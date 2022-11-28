"""
A processor that encrypts model weights in MaskCrypt.
"""
import asyncio
import logging
import pickle
import re
import sys
import uuid
from typing import Any

import torch
import socketio
from plato.processors import model
from plato.callbacks.handler import CallbackHandler
from plato.callbacks.client import LogProgressCallback
from plato.config import Config

class MPCClientEvents(socketio.AsyncClientNamespace):
    """A custom namespace for socketio.AsyncServer."""

    def __init__(self, namespace, mpc_client):
        super().__init__(namespace)
        self.mpc_client = mpc_client
        self.client_id = mpc_client.client_id

    async def on_connect(self):
        """Upon a new connection to the server."""
        logging.info("[Client #%d] Connected to the relay server.", self.client_id)

    async def on_disconnect(self):
        """Upon a disconnection event."""
        logging.info(
            "[Client #%d] The relay server disconnected the connection.", self.client_id
        )
        #self.mpc_client._clear_checkpoint_files()

    async def on_connect_error(self, data):
        """Upon a failed connection attempt to the server."""
        logging.info(
            "[Client #%d] A connection attempt to the relay server failed.", self.client_id
        )

    # async def on_payload_to_arrive(self, data):
    #     """New payload is about to arrive from the relay server."""
    #     await self.mpc_client._payload_to_arrive(data["response"])

    # async def on_request_update(self, data):
    #     """The relay server is requesting an urgent model update."""
    #     await self.mpc_client._request_update(data)

    # async def on_chunk(self, data):
    #     """A chunk of data from the relay server arrived."""
    #     await self.mpc_client._chunk_arrived(data["data"])

    # async def on_payload(self, data):
    #     """A portion of the new payload from the relay server arrived."""
    #     await self.mpc_client._payload_arrived(data["id"])

    # async def on_payload_done(self, data):
    #     """All of the new payload sent from the relay server arrived."""
    #     if "s3_key" in data:
    #         await self.mpc_client._payload_done(data["id"], s3_key=data["s3_key"])
    #     else:
    #         await self.mpc_client._payload_done(data["id"])

    #todo: handle receive


class Processor(model.Processor):
    """
    A processor that encrypts model weights with given encryption mask.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        asyncio.ensure_future(self._init_mpc_client())

    async def _init_mpc_client(self):
        """Initialize client of the relay server."""
        logging.info("[MPC Client #%d] Contacting the server.", self.client_id)

        self.sio = socketio.AsyncClient(reconnection=True)
        self.sio.register_namespace(MPCClientEvents(namespace="/relay", mpc_client=self))

        if hasattr(Config().relay_server, "use_https"):
            uri = f"https://{Config().relay_server.address}"
        else:
            uri = f"http://{Config().relay_server.address}"

        if hasattr(Config().relay_server, "port"):
            uri = f"{uri}:{Config().relay_server.port}"

        logging.info("[%s] MPC client connecting to the relay server at %s.", self, uri)
        await self.sio.connect(uri, wait_timeout=600)
        await self.sio.emit("client_alive", {"id": self.client_id})

        logging.info("[MPC Client #%d] is connected.", self.client_id)
        await self.sio.wait() #todo: wait for which event?

    async def _send_in_chunks(self, data) -> None:
        """Sending a bytes object in fixed-sized chunks to the client."""
        step = 1024 ^ 2
        chunks = [data[i : i + step] for i in range(0, len(data), step)]

        for chunk in chunks:
            await self.sio.emit("chunk", {"data": chunk})

        await self.sio.emit("client_payload", {"id": self.client_id}) #todo: to_client_id

    async def _send(self, payload) -> None:
        """Sending the client payload to the server using simulation, S3 or socket.io."""
        #todo: process comm_simulation data
        # if self.comm_simulation:
        #     pass
        # else:
        metadata = {"id": self.client_id}

        #todo: process s3 storage
        # if self.s3_client is not None:
        #     unique_key = uuid.uuid4().hex[:6].upper()
        #     s3_key = f"client_payload_{self.client_id}_{unique_key}"
        #     self.s3_client.send_to_s3(s3_key, payload)
        #     data_size = sys.getsizeof(pickle.dumps(payload))
        #     metadata["s3_key"] = s3_key
        # else:
        if isinstance(payload, list):
            data_size: int = 0

            for data in payload:
                _data = pickle.dumps(data)
                await self._send_in_chunks(_data)
                data_size += sys.getsizeof(_data)
        else:
            _data = pickle.dumps(payload)
            await self._send_in_chunks(_data)
            data_size = sys.getsizeof(_data)

        await self.sio.emit("client_payload_done", metadata)

        logging.info(
            "[%s] Sent %.2f MB of payload data to the server.",
            self,
            data_size / 1024**2,
        )

    def process(self, data: Any) -> Any:
        logging.info(
            "[Client #%d] Encrypt the model weights with given encryption mask.",
            self.client_id,
        )

        print("in_outbound_process")
        print(data)

        asyncio.ensure_future(self._send(data))

        return data
