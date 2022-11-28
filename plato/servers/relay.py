"""
A simple federated learning server using federated averaging.
"""

import asyncio
import socketio
import logging
import os

from aiohttp import web

from plato.algorithms import registry as algorithms_registry
from plato.config import Config
from plato.processors import registry as processor_registry
from plato.servers import base
from plato.trainers import registry as trainers_registry
from plato.utils import csv_processor, fonts

class RelayServerEvents(socketio.AsyncNamespace):
    """A custom namespace for socketio.AsyncServer."""

    def __init__(self, namespace, relay_server):
        super().__init__(namespace)
        self.relay_server = relay_server

    async def on_connect(self, sid, environ):
        """Upon a new connection from a client."""
        logging.info("[Server #%d] A new client just connected.", os.getpid())

    async def on_disconnect(self, sid):
        """Upon a disconnection event."""
        logging.info("[Server #%d] An existing client just disconnected.", os.getpid())
        # await self.relay_server._client_disconnected(sid)

    async def on_client_alive(self, sid, data):
        """A new client arrived or an existing client sends a heartbeat."""
        # await self.relay_server.register_client(sid, data["id"])

    async def on_client_report(self, sid, data):
        """An existing client sends a new report from local training."""
        # await self.relay_server._client_report_arrived(sid, data["id"], data["report"])

    async def on_chunk(self, sid, data):
        """A chunk of data from the server arrived."""
        # await self.relay_server._client_chunk_arrived(sid, data["data"])

    async def on_client_payload(self, sid, data):
        """An existing client sends a new payload from local training."""
        # await self.relay_server._client_payload_arrived(sid, data["id"])

    async def on_client_payload_done(self, sid, data):
        """An existing client finished sending its payloads from local training."""
        # if "s3_key" in data:
        #     await self.relay_server._client_payload_done(
        #         sid, data["id"], s3_key=data["s3_key"]
        #     )
        # else:
        #     await self.relay_server._client_payload_done(sid, data["id"])


class Server(base.Server):
    """Federated learning server using federated averaging."""

    def __init__(
        self, callbacks=None
    ):
        super().__init__(callbacks=callbacks)

        self.total_clients = Config().clients.total_clients
        self.clients_per_round = Config().clients.per_round

        logging.info(
            "[Relay Server #%d] Started training on %d clients with %d per round.",
            os.getpid(),
            self.total_clients,
            self.clients_per_round,
        )
       
    def run(self, client=None, edge_server=None, edge_client=None, trainer=None):
        """Start a run loop for the server."""
        self.client = client
        # self.configure()

        # if Config().args.resume:
        #     self._resume_from_checkpoint()

        asyncio.get_event_loop().create_task(self._periodic(self.periodic_interval)) 
        port = Config().relay_server.port
        host = Config().relay_server.address
        print("port, host: ", port, host)
        # super().start(Config().relay_server.port, Config().relay_server.address)
        logging.info(
            "Starting a server at address %s and port %s.",
            host,
            port,
        )

        self.sio = socketio.AsyncServer(
            ping_interval=self.ping_interval,
            max_http_buffer_size=2**31,
            ping_timeout=self.ping_timeout,
        )
        self.sio.register_namespace(RelayServerEvents(namespace="/", relay_server=self))

        # if hasattr(Config().server, "s3_endpoint_url"):
        #     self.s3_client = s3.S3()

        app = web.Application()
        self.sio.attach(app)
        web.run_app(
            app, host=host, port=port, loop=asyncio.get_event_loop()
        )

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]

        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # Extract the current model weights as the baseline
        baseline_weights = self.algorithm.extract_weights()

        if hasattr(self, "aggregate_weights"):
            # Runs a server aggregation algorithm using weights rather than deltas
            logging.info(
                "[Server #%d] Aggregating model weights directly rather than weight deltas.",
                os.getpid(),
            )

            updated_weights = await self.aggregate_weights(
                self.updates, baseline_weights, weights_received
            )

            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)
        else:
            # Computes the weight deltas by comparing the weights received with
            # the current global model weights
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            # Runs a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[Server #%d] Aggregating model weight deltas.", os.getpid())
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)

        # The model weights have already been aggregated, now calls the
        # corresponding hook and callback
        self.weights_aggregated(self.updates)
        self.callback_handler.call_event("on_weights_aggregated", self, self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server
            logging.info("[%s] Started model testing.", self)
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model perplexity: {self.accuracy:.2f}\n"
                )
            )
        else:
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n"
                )
            )

        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)
    

    def on_weights_received(self, server, weights_received) :
        """todo: Override this method to complete additional tasks after the updated weights have been received."""
        print("on_weights_received: weights are received")
        print(weights_received.size())

    def customize_server_response(self, server_response: dict, client_id) -> None:
        """todo: customize server response with any additional information."""
    
    def clients_processed(self) -> None:
        """Additional work to be performed after client reports have been processed."""

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        return {
            "round": self.current_round,
            "accuracy": self.accuracy,
            "elapsed_time": self.wall_time - self.initial_wall_time,
            "comm_time": max(update.report.comm_time for update in self.updates),
            "round_time": max(
                update.report.training_time + update.report.comm_time
                for update in self.updates
            ),
            "comm_overhead": self.comm_overhead,
        }




