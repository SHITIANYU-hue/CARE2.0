"""Process-local fail-closed network guard for offline preparation."""
import socket
def denied(*args,**kwargs): raise RuntimeError("AUC reanalysis: all network access is disabled")
socket.socket.connect=denied
socket.socket.connect_ex=denied
socket.socket.sendto=denied
socket.create_connection=denied
socket.getaddrinfo=denied
