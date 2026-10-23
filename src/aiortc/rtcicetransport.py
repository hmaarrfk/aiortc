import asyncio
import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass
from typing import Any, Optional

from aioice import Candidate, Connection, ConnectionClosed, TransportPolicy
from aioice import ice as aioice_ice
from pyee.asyncio import AsyncIOEventEmitter

from .exceptions import InvalidStateError
from .rtcconfiguration import RTCIceServer

# aioice gathers server-reflexive (STUN) and relay (TURN) candidates from every
# local interface and waits, ALL_COMPLETED, up to a hardcoded 5 seconds. An
# interface that cannot reach the STUN/TURN server -- VPNs such as tailscale,
# virtual bridges, link-local NICs -- never gets a reply, so its task never
# completes and gathering stalls for the full 5s on every connection, even
# though the usable candidates arrive in well under a second. For an answerer
# (recvonly media server) this delay is the dominant connection-setup latency.
#
# Cap the per-component gather with a much shorter, configurable timeout. The
# candidates that are going to succeed do so almost immediately; the timeout
# only bounds how long we wait on interfaces that will never answer.
ICE_GATHER_TIMEOUT = float(os.environ.get("AIORTC_ICE_GATHER_TIMEOUT", "1.0"))
# Once the relay candidate and one server-reflexive candidate are in, the
# remaining requests get this long to answer before gathering returns without
# them. A request that answers at all answers within a round trip of the
# first one, so the timeout above is only reached when TURN is unreachable.
ICE_GATHER_GRACE = float(os.environ.get("AIORTC_ICE_GATHER_GRACE", "0.1"))


class _ShortGatherConnection(Connection):
    """aioice ``Connection`` whose reflexive/relay gathering returns as soon as
    the useful candidates are in rather than waiting on every interface. See
    ``ICE_GATHER_TIMEOUT`` and ``ICE_GATHER_GRACE``."""

    async def get_component_candidates(
        self, component: int, addresses: list[str], timeout: float = ICE_GATHER_TIMEOUT
    ) -> list[Candidate]:
        # aioice's own implementation, up to the wait on the STUN and TURN
        # tasks: it waits for all of them, so one interface that never hears
        # back holds every connection for the whole timeout.
        candidates = []
        loop = asyncio.get_running_loop()

        host_protocols = []
        for address in addresses:
            try:
                transport, protocol = await loop.create_datagram_endpoint(
                    lambda: aioice_ice.StunProtocol(self), local_addr=(address, 0)
                )
                sock = transport.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(
                        socket.SOL_SOCKET,
                        socket.SO_RCVBUF,
                        aioice_ice.turn.UDP_SOCKET_BUFFER_SIZE,
                    )
            except OSError as exc:
                logger.info("Could not bind to %s - %s", address, exc)
                continue
            host_protocols.append(protocol)

            candidate_address = protocol.transport.get_extra_info("sockname")
            protocol.local_candidate = Candidate(
                foundation=aioice_ice.candidate_foundation(
                    "host", "udp", candidate_address[0]
                ),
                component=component,
                transport="udp",
                priority=aioice_ice.candidate_priority(component, "host"),
                host=candidate_address[0],
                port=candidate_address[1],
                type="host",
            )
            if self._transport_policy == TransportPolicy.ALL:
                candidates.append(protocol.local_candidate)
        self._protocols += host_protocols

        reflexive_tasks = []
        if self.stun_server:
            for protocol in host_protocols:
                if ipaddress.ip_address(protocol.local_candidate.host).version == 4:
                    reflexive_tasks.append(
                        asyncio.create_task(
                            aioice_ice.server_reflexive_candidate(
                                protocol, self.stun_server
                            )
                        )
                    )

        relay_task = None
        if self.turn_server:
            relay_task = asyncio.create_task(
                aioice_ice.relayed_candidate(
                    component=component,
                    protocol_factory=lambda: aioice_ice.StunProtocol(self),
                    turn_server=self.turn_server,
                    turn_username=self.turn_username,
                    turn_password=self.turn_password,
                    turn_ssl=self.turn_ssl,
                    turn_transport=self.turn_transport,
                )
            )

        tasks = reflexive_tasks + ([relay_task] if relay_task else [])
        if tasks:
            await _wait_for_useful_candidates(
                reflexive_tasks, relay_task, timeout, ICE_GATHER_GRACE
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
                elif task.exception() is None:
                    candidate, protocol = task.result()
                    candidates.append(candidate)
                    if protocol is not None:
                        self._protocols.append(protocol)

        return candidates


def _succeeded(task: asyncio.Task) -> bool:
    return task.done() and not task.cancelled() and task.exception() is None


async def _wait_for_useful_candidates(
    reflexive_tasks: list[asyncio.Task],
    relay_task: Optional[asyncio.Task],
    timeout: float,
    grace: float,
) -> None:
    """Wait until the relay request has finished and one reflexive request has
    succeeded (or all of them have finished), then give the rest ``grace``
    seconds, never waiting longer than ``timeout`` in all."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    pending = set(reflexive_tasks)
    if relay_task is not None:
        pending.add(relay_task)

    def useful() -> bool:
        relay_settled = relay_task is None or relay_task.done()
        reflexive_settled = all(task.done() for task in reflexive_tasks) or any(
            _succeeded(task) for task in reflexive_tasks
        )
        return relay_settled and reflexive_settled

    while pending and not useful():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        _, pending = await asyncio.wait(
            pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
        )
    remaining = min(grace, deadline - loop.time())
    if pending and remaining > 0:
        await asyncio.wait(pending, timeout=remaining)


# See https://datatracker.ietf.org/doc/html/rfc7064
# transport is not defined by RFC 7064 and rejected by browsers.
STUN_REGEX = re.compile(
    r"(?P<scheme>stun|stuns)\:(?P<host>[^?:]+)(\:(?P<port>[0-9]+?))?"
    r"(\?transport=(?P<transport>.*))?"
)
# See https://datatracker.ietf.org/doc/html/rfc7065
TURN_REGEX = re.compile(
    r"(?P<scheme>turn|turns)\:(?P<host>[^?:]+)(\:(?P<port>[0-9]+?))?"
    r"(\?transport=(?P<transport>.*))?"
)

logger = logging.getLogger(__name__)


@dataclass
class RTCIceCandidate:
    """
    The :class:`RTCIceCandidate` interface represents a candidate Interactive
    Connectivity Establishment (ICE) configuration which may be used to
    establish an RTCPeerConnection.
    """

    component: int
    foundation: str
    ip: str
    port: int
    priority: int
    protocol: str
    type: str
    relatedAddress: Optional[str] = None
    relatedPort: Optional[int] = None
    sdpMid: Optional[str] = None
    sdpMLineIndex: Optional[int] = None
    tcpType: Optional[str] = None


@dataclass
class RTCIceParameters:
    """
    The :class:`RTCIceParameters` dictionary includes the ICE username
    fragment and password and other ICE-related parameters.
    """

    usernameFragment: Optional[str] = None
    "ICE username fragment."

    password: Optional[str] = None
    "ICE password."

    iceLite: bool = False


def candidate_from_aioice(x: Candidate) -> RTCIceCandidate:
    return RTCIceCandidate(
        component=x.component,
        foundation=x.foundation,
        ip=x.host,
        port=x.port,
        priority=x.priority,
        protocol=x.transport,
        relatedAddress=x.related_address,
        relatedPort=x.related_port,
        tcpType=x.tcptype,
        type=x.type,
    )


def candidate_to_aioice(x: RTCIceCandidate) -> Candidate:
    return Candidate(
        component=x.component,
        foundation=x.foundation,
        host=x.ip,
        port=x.port,
        priority=x.priority,
        related_address=x.relatedAddress,
        related_port=x.relatedPort,
        transport=x.protocol,
        tcptype=x.tcpType,
        type=x.type,
    )


def connection_kwargs(servers: list[RTCIceServer]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    for server in servers:
        if isinstance(server.urls, list):
            uris = server.urls
        else:
            uris = [server.urls]

        for uri in uris:
            parsed = parse_stun_turn_uri(uri)

            if parsed["scheme"] == "stun":
                # only a single STUN server is supported
                if "stun_server" in kwargs:
                    continue

                kwargs["stun_server"] = (parsed["host"], parsed["port"])
            elif parsed["scheme"] in ["turn", "turns"]:
                # only a single TURN server is supported
                if "turn_server" in kwargs:
                    continue

                # only 'udp' and 'tcp' transports are supported
                if parsed["scheme"] == "turn" and parsed["transport"] not in [
                    "udp",
                    "tcp",
                ]:
                    continue
                elif parsed["scheme"] == "turns" and parsed["transport"] != "tcp":
                    continue

                # only 'password' credentialType is supported
                if server.credentialType != "password":
                    continue

                kwargs["turn_server"] = (parsed["host"], parsed["port"])
                kwargs["turn_ssl"] = parsed["scheme"] == "turns"
                kwargs["turn_transport"] = parsed["transport"]
                kwargs["turn_username"] = server.username
                kwargs["turn_password"] = server.credential

    return kwargs


def parse_stun_turn_uri(uri: str) -> dict[str, Any]:
    if uri.startswith("stun"):
        match = STUN_REGEX.fullmatch(uri)
    elif uri.startswith("turn"):
        match = TURN_REGEX.fullmatch(uri)
    else:
        raise ValueError("malformed uri: invalid scheme")

    if not match:
        raise ValueError("malformed uri")

    # set port
    parsed: dict[str, Any] = match.groupdict()
    if parsed["port"]:
        parsed["port"] = int(parsed["port"])
    elif parsed["scheme"] in ["stuns", "turns"]:
        parsed["port"] = 5349
    else:
        parsed["port"] = 3478

    # set transport
    if parsed["scheme"] == "turn" and not parsed["transport"]:
        parsed["transport"] = "udp"
    elif parsed["scheme"] == "turns" and not parsed["transport"]:
        parsed["transport"] = "tcp"
    elif parsed["scheme"] in ["stun", "stuns"]:
        if parsed["transport"] is not None:
            raise ValueError(
                "malformed uri: " + parsed["scheme"] + " must not contain transport"
            )
        del parsed["transport"]

    return parsed


class RTCIceGatherer(AsyncIOEventEmitter):
    """
    The :class:`RTCIceGatherer` interface gathers local host, server reflexive
    and relay candidates, as well as enabling the retrieval of local
    Interactive Connectivity Establishment (ICE) parameters which can be
    exchanged in signaling.
    """

    def __init__(
        self,
        iceServers: Optional[list[RTCIceServer]] = None,
        local_username: Optional[str] = None,
        local_password: Optional[str] = None,
    ) -> None:
        super().__init__()

        if iceServers is None:
            iceServers = self.getDefaultIceServers()
        ice_kwargs = connection_kwargs(iceServers)

        self._connection = _ShortGatherConnection(
            ice_controlling=False,
            local_username=local_username,
            local_password=local_password,
            **ice_kwargs,
        )
        self._remote_candidates_end = False
        self.__state = "new"

    @property
    def state(self) -> str:
        """
        The current state of the ICE gatherer.
        """
        return self.__state

    async def gather(self) -> None:
        """
        Gather ICE candidates.
        """
        if self.__state == "new":
            self.__setState("gathering")
            await self._connection.gather_candidates()
            self.__setState("completed")

    @classmethod
    def getDefaultIceServers(cls) -> list[RTCIceServer]:
        """
        Return the list of default :class:`RTCIceServer`.
        """
        return [RTCIceServer("stun:stun.l.google.com:19302")]

    def getLocalCandidates(self) -> list[RTCIceCandidate]:
        """
        Retrieve the list of valid local candidates associated with the ICE
        gatherer.
        """
        return [candidate_from_aioice(x) for x in self._connection.local_candidates]

    def getLocalParameters(self) -> RTCIceParameters:
        """
        Retrieve the ICE parameters of the ICE gatherer.

        :rtype: RTCIceParameters
        """
        return RTCIceParameters(
            usernameFragment=self._connection.local_username,
            password=self._connection.local_password,
        )

    def __setState(self, state: str) -> None:
        self.__state = state
        self.emit("statechange")


class RTCIceTransport(AsyncIOEventEmitter):
    """
    The :class:`RTCIceTransport` interface allows an application access to
    information about the Interactive Connectivity Establishment (ICE)
    transport over which packets are sent and received.

    :param gatherer: An :class:`RTCIceGatherer`.
    """

    def __init__(self, gatherer: RTCIceGatherer) -> None:
        super().__init__()
        self.__iceGatherer = gatherer
        self.__monitor_task: Optional[asyncio.Future[None]] = None
        self.__start: Optional[asyncio.Event] = None
        self.__state = "new"
        self._connection = gatherer._connection
        self._role_set = False

        # expose recv / send methods
        self._recv = self._connection.recv
        self._send = self._connection.send

    @property
    def iceGatherer(self) -> RTCIceGatherer:
        """
        The ICE gatherer passed in the constructor.
        """
        return self.__iceGatherer

    @property
    def role(self) -> str:
        """
        The current role of the ICE transport.

        Either `'controlling'` or `'controlled'`.
        """
        if self._connection.ice_controlling:
            return "controlling"
        else:
            return "controlled"

    @property
    def state(self) -> str:
        """
        The current state of the ICE transport.
        """
        return self.__state

    async def addRemoteCandidate(self, candidate: Optional[RTCIceCandidate]) -> None:
        """
        Add a remote candidate.

        :param candidate: The new candidate or `None` to signal end of candidates.
        """
        if not self.__iceGatherer._remote_candidates_end:
            if candidate is None:
                self.__iceGatherer._remote_candidates_end = True
                await self._connection.add_remote_candidate(None)
            else:
                await self._connection.add_remote_candidate(
                    candidate_to_aioice(candidate)
                )

    def getRemoteCandidates(self) -> list[RTCIceCandidate]:
        """
        Retrieve the list of candidates associated with the remote
        :class:`RTCIceTransport`.
        """
        return [candidate_from_aioice(x) for x in self._connection.remote_candidates]

    async def start(self, remoteParameters: RTCIceParameters) -> None:
        """
        Initiate connectivity checks.

        :param remoteParameters: The :class:`RTCIceParameters` associated with
                                  the remote :class:`RTCIceTransport`.
        """
        if self.state == "closed":
            raise InvalidStateError("RTCIceTransport is closed")

        # handle the case where start is already in progress
        if self.__start is not None:
            await self.__start.wait()
            return
        self.__start = asyncio.Event()
        self.__monitor_task = asyncio.ensure_future(self._monitor())

        self.__setState("checking")
        self._connection.remote_is_lite = remoteParameters.iceLite
        self._connection.remote_username = remoteParameters.usernameFragment
        self._connection.remote_password = remoteParameters.password
        try:
            await self._connection.connect()
        except ConnectionError:
            self.__setState("failed")
        else:
            self.__setState("completed")
        self.__start.set()

    async def stop(self) -> None:
        """
        Irreversibly stop the :class:`RTCIceTransport`.
        """
        if self.state != "closed":
            self.__setState("closed")
            await self._connection.close()
            if self.__monitor_task is not None:
                await self.__monitor_task
                self.__monitor_task = None

    async def _monitor(self) -> None:
        while True:
            event = await self._connection.get_event()
            if isinstance(event, ConnectionClosed):
                if self.state == "completed":
                    self.__setState("failed")
                return

    def __log_debug(self, msg: str, *args: object) -> None:
        logger.debug(f"RTCIceTransport(%s) {msg}", self.role, *args)

    def __setState(self, state: str) -> None:
        if state != self.__state:
            self.__log_debug("- %s -> %s", self.__state, state)
            self.__state = state
            self.emit("statechange")

            # no more events will be emitted, so remove all event listeners
            # to facilitate garbage collection.
            if state == "closed":
                self.iceGatherer.remove_all_listeners()
                self.remove_all_listeners()
