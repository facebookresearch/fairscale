# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Bi-Distributed Gossip Wrapper

:description: Multi-Threaded Bipartite-Distributed Gossip Wrapper, designed
              for efficient multi-peer training using bipartite
              agent roles, with asynchronous passive nodes,
              synchronous active nodes, and push-pull
              communication.
"""

import os
import time
import torch.multiprocessing as mp

import torch
import torch.distributed as dist
from torch.cuda.comm import broadcast_coalesced, reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply

from .gossiper import BilatPushPull
from .utils import make_logger, flatten_tensors, unflatten_tensors
from .utils.metering import Meter


class BilatGossipDataParallel(Module):
    """ Distributed Gossip model wrapper """

    def __init__(self, module, device_ids=None, master_addr=None,
                 master_port=None, backend=None, world_size=None, rank=None,
                 graph_class=None, mixing_class=None, num_peers=1,
                 comm_device=None, lr=0.1, momentum=0.9, weight_decay=1e-4,
                 nesterov=True, verbose=True, network_interface_type=None,
                 tcp_interface_name=None):
        super(BilatGossipDataParallel, self).__init__()

        # devices available locally
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.output_device = device_ids[0]
        self.device_ids = device_ids

        # put model on output device
        self.module = module.cuda(self.output_device)

        # prepare local intra-node all-reduce objects
        if len(self.device_ids) > 1:
            self.broadcast_bucket_size = 10 * 1024 * 1024  # bytes
            self.nccl_reduce_bucket_size = 256 * 1024 * 1024  # bytes

            self._module_copies = replicate(self.module, self.device_ids,
                                            detach=True)
            self._module_copies[0] = self.module
            for cmodule in self._module_copies[1:]:
                for p, cp in zip(self.module.parameters(),
                                 cmodule.parameters()):
                    cp.requires_grad = p.requires_grad
        else:
            self._module_copies = [self.module]

        # communicate over cpu's if not specified
        if comm_device is None:
            comm_device = torch.device('cpu')
        self.__cpu_comm = comm_device.type == 'cpu'

        # distributed backend config
        self.dist_config = {
            'verbose': verbose,
            'graph_class': graph_class,
            'master_addr': master_addr,
            'master_port': master_port,
            'backend': backend,
            'world_size': world_size,
            'rank': rank,
            'mixing_class': mixing_class,
            'lr': lr,
            'momentum': momentum,
            'nesterov': nesterov,
            'weight_decay': weight_decay,
            'comm_device': comm_device,
            'network_interface_type': network_interface_type,
            'num_peers': num_peers
        }
        self.num_updates = 0

        # logger used to print to stdout
        self.logger = make_logger(rank, verbose)

        # prepare parameters for gossip
        self.gossip_enable = True
        self.gossip_params = []
        self.gossip_grads = []
        for p in module.parameters():
            cp = p.clone().detach_()
            cp = cp.cpu().pin_memory() if self.__cpu_comm else cp.cuda()
            cp.requires_grad = p.requires_grad
            self.gossip_params.append(cp)
            if p.requires_grad:
                g = cp.clone().zero_().detach_()
                g = g.cpu().pin_memory() if self.__cpu_comm else g.cuda()
                self.gossip_grads.append(g)

        self.gossip_queue = mp.Queue()
        self.gossip_lock = mp.Lock()
        self.gossip_enable_flag = mp.Event()
        self.train_write_flag = mp.Event()  # signal train-proc write event
        self.gossip_read_flag = mp.Event()  # signal gossip-proc read event
        self.gossip_update_flag = mp.Event()  # signal 2 gossip-proc need update
        self._lr = mp.Value('f', lr, lock=self.gossip_lock)
        self.gossip_thread = mp.Process(
            target=BilatGossipDataParallel._gossip_target,
            args=(self.dist_config,
                  self.gossip_enable_flag,
                  self.train_write_flag,
                  self.gossip_read_flag,
                  self.gossip_update_flag,
                  self._lr,
                  self.gossip_lock,
                  self.gossip_queue,
                  tcp_interface_name))
        self.gossip_thread.daemon = True
        self.gossip_thread.name = 'Gossip-Thread'
        self.gossip_thread.start()

        # pass handle to gossip_params and gossip_grads, and put in shared
        # memory
        self.gossip_queue.put((self.gossip_params, self.gossip_grads))

        # register ps/grad-reduction hooks
        self.__register_hooks()

    def update_lr(self, lr):
        if self._lr.value == lr:
            return
        self._lr.value = lr
        self.gossip_update_flag.set()

    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) > 1:
            # run forward pass across all devices
            self._sync_params()
            outputs = self.parallel_apply(self._module_copies[:len(inputs)],
                                          inputs, kwargs)
            return self.gather(outputs, self.output_device)
        else:
            return self.module(*inputs[0], **kwargs[0])

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs,
                              self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=0)

    def _sync_params(self):
        """ Synchronize parameters across devices (intra-node) """
        if len(self.device_ids) <= 1:
            return

        # intra-node parameter sync
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(params, self.device_ids,
                                     self.broadcast_bucket_size)
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                param.data.set_(tensor)

        # intra-node buffer sync
        buffers = [b.data for b in self.module.buffers()]
        if len(buffers) > 0:
            result = broadcast_coalesced(buffers, self.device_ids,
                                         self.broadcast_bucket_size)
            for tensors, module in zip(result[1:], self._module_copies[1:]):
                for tensor, buf in zip(tensors, module.buffers()):
                    buf.data.set_(tensor)

    def train(self, mode=True):
        super(BilatGossipDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(BilatGossipDataParallel, self).eval()
        for module in self._module_copies[1:]:
            module.eval()
        self._pull_model()

    def enable_gossip(self):
        self.gossip_enable = True
        self.gossip_enable_flag.set()

    def disable_gossip(self):
        self.gossip_enable = False
        self.gossip_enable_flag.clear()

    def block(self):
        return
        self.logger.info('blocking')
        dist.barrier()

    def sync_comms(self):
        self._pull_model()

    def _pull_model(self):
        """ Pull model from gossip thread """

        # pull model from gossip thread
        with self.gossip_lock:
            for p, r in zip(self.module.parameters(), self.gossip_params):
                p.data.copy_(r, non_blocking=False)

        # update flags
        self.logger.debug('updated model params')
        return True

    def _transfer_grads(self):
        """ Transfers copy of grads to gossip thread """
        self.logger.debug('transfering model grads')

        # don't transfer new grads until old grads have been transferred
        self.gossip_read_flag.wait()

        # transfer the new grads
        i = 0
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                self.gossip_grads[i].copy_(p.grad, non_blocking=False)
                i += 1

        # update flags
        self.logger.debug('transfered model grads')
        self.gossip_read_flag.clear()
        self.train_write_flag.set()
        return True

    @staticmethod
    def _gossip_target(dist_config, gossip_enable_flag, train_write_flag,
                       gossip_read_flag, gossip_update_flag, gossip_lr,
                       gossip_lock, gossip_queue, tcp_interface_name):
        """ Gossip thread, which performs push-sum on model params """
        with torch.no_grad():
            gossip_params, gossip_grads = gossip_queue.get()

            # prepare gossip process control objects
            gossip_optimizer = torch.optim.SGD(
                gossip_params,
                lr=dist_config['lr'],
                momentum=dist_config['momentum'],
                weight_decay=dist_config['weight_decay'],
                nesterov=dist_config['nesterov'])

            if dist_config['backend'] == 'gloo':
                assert dist_config['network_interface_type'] == 'ethernet'
            elif dist_config['network_interface_type'] == 'ethernet':
                if dist_config['backend'] == 'nccl':
                    os.environ['NCCL_SOCKET_IFNAME'] = tcp_interface_name
                    os.environ['NCCL_IB_DISABLE'] = '1'
                elif dist_config['backend'] == 'gloo':
                    os.environ['GLOO_SOCKET_IFNAME'] = tcp_interface_name
                else:
                    raise NotImplementedError

            # initialize torch distributed backend
            os.environ['MASTER_ADDR'] = dist_config['master_addr']
            os.environ['MASTER_PORT'] = dist_config['master_port']
            dist.init_process_group(backend=dist_config['backend'],
                                    world_size=dist_config['world_size'],
                                    rank=dist_config['rank'])

            logger = make_logger(dist.get_rank(), dist_config['verbose'])
            logger.debug('init rcvd: gossip_params {}, gossip_grads {}'.format(
                gossip_params[0].norm(), gossip_grads[0].norm()))

            # init gossip instance
            graph_class = dist_config['graph_class']
            mixing_class = dist_config['mixing_class']

            if graph_class:
                # dist.barrier is done here to ensure the NCCL communicator is
                # created here. This prevents an error which may be caused if
                # the NCCL # communicator is created at a time gap of more
                # than 5 minutes in different processes
                dist.barrier()
                graph = graph_class(
                    dist_config['rank'], dist_config['world_size'],
                    peers_per_itr=dist_config['num_peers'])
            if mixing_class and graph:
                mixing = mixing_class(graph, dist_config['comm_device'])

            gossiper = BilatPushPull(flatten_tensors(gossip_params),
                                     graph=graph,
                                     mixing=mixing,
                                     logger=logger)

            dist_config['graph'] = gossiper._graph_manager
            dist_config['mixing'] = gossiper._mixing_manager
            dist_config['gossiper'] = gossiper
            model_meter = Meter(ptag='Model', stateful=True, csv_format=False)
            gossip_meter = Meter(
                ptag='Gossip', stateful=True, csv_format=False)
            gossip_read_flag.set()

            # gossip loop
            while True:
                # we may be asked to hold off on gossip for some time
                gossip_enable_flag.wait()

                # we may be notified to update our learning rate
                if gossip_update_flag.is_set():
                    for pg in gossip_optimizer.param_groups:
                        pg['lr'] = gossip_lr.value
                    logger.debug('updated lr to {}'.format(gossip_lr.value))
                    gossip_update_flag.clear()

                # train process is telling us it computed the new grads
                if train_write_flag.is_set():
                    bt = time.time()
                    with gossip_lock:
                        i = 0
                        for p in gossip_params:
                            if p.requires_grad:
                                p.grad = gossip_grads[i]
                                i += 1
                        gossip_optimizer.step()
                        gossip_optimizer.zero_grad()
                    train_write_flag.clear()
                    gossip_read_flag.set()
                    model_meter.update(time.time() - bt)
                    logger.debug(model_meter)

                try:
                    # construct gossip tensor
                    bt = time.time()
                    with gossip_lock:
                        out_msg = flatten_tensors(gossip_params).to(
                            dist_config['comm_device'])
                    # gossip step
                    in_msg, completed = gossiper.mix(out_msg)
                    # update gossip params (local model)
                    if completed:
                        with gossip_lock:
                            for p, g in zip(
                                    gossip_params, unflatten_tensors(
                                        in_msg, gossip_params)):
                                p.data.add_(g.to(p.device)).mul_(0.5)
                    gossip_meter.update(time.time() - bt)
                    logger.debug(gossip_meter)
                except RuntimeError as e:
                    logger.warning('received runtime error {}'.format(e))
                    gossiper.clean_msg_buffers_()

    def __register_hooks(self):
        """
        Registers push-sum de-bias/bias hooks in pre-forward/post-backward
        passes in all leaf modules
        """
        self.register_backward_hook(self.__make_backward_hook())

    def __make_backward_hook(self):
        self.logger.debug('making backward hook')

        def hook(*unused):
            # reduce gradients across devices on a single machine
            if len(self.device_ids) > 1:

                # collect gradients from all copies
                all_grads = [[] for _ in range(len(self._module_copies))]
                for dev_idx, module in enumerate(self._module_copies):
                    for p in module.parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        all_grads[dev_idx].append(p.grad.data)

                # reduce grads
                reduced_grads = reduce_add_coalesced(
                    all_grads, self.output_device,
                    self.nccl_reduce_bucket_size)

                # update grads with reduced grads
                for grad, reduced in zip(all_grads[0], reduced_grads):
                    grad.copy_(reduced)

                # clear the gradients and parameters across all replicas
                for module in self._module_copies[1:]:
                    for param in module.parameters():
                        if param.requires_grad:
                            param.grad = None
                            param.data.set_()

            # convert model back to ps-numerator
            self._transfer_grads()
            self._pull_model()

        def queue_hook(*unused):
            Variable._execution_engine.queue_callback(hook)
        return queue_hook

    def communicator_warmup(self):
        """ time the all-reducde code """
        dist.barrier()
        time.sleep(5)
        dist.barrier()
