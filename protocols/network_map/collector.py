"""Network map collector: read the ARP table and the default gateway.

Linux:   /proc/net/arp and /proc/net/route  (plain file reads, no commands)
Windows: `arp -a` and `route print -4`     (two allow-listed commands)

Nothing here sends a single packet. The ARP table is a cache the operating
system fills by itself while you use the network; we just read it.

Parsing is done with patterns that look for IP and MAC address shapes rather
than exact header words, so it also works on non-English Windows
("Schnittstelle", "dynamisch", ...).
"""

from __future__ import annotations

import re

IP = r"\d{1,3}(?:\.\d{1,3}){3}"
MAC_ANY = r"[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}"

# Linux ARP flags (from the kernel): 0x2 = complete, 0x4 = permanent (static)
ATF_COM = 0x2
ATF_PERM = 0x4
RTF_GATEWAY = 0x2


def parse_linux_arp(text: str) -> list[dict]:
    """/proc/net/arp:
    IP address  HW type  Flags  HW address         Mask  Device
    192.168.1.1 0x1      0x2    a4:2b:b0:11:22:33  *     wlan0
    """
    entries = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6 or not re.fullmatch(IP, parts[0]):
            continue
        try:
            flags = int(parts[2], 16)
        except ValueError:
            continue
        entries.append({
            "ip": parts[0],
            "mac": parts[3].lower(),
            "interface": parts[5],
            "complete": bool(flags & ATF_COM),
            "type": "static" if flags & ATF_PERM else "dynamic",
        })
    return entries


def _hex_le_to_ip(hex_text: str) -> str | None:
    """/proc/net/route stores IPv4 addresses as little-endian hex:
    '0101A8C0' -> bytes C0 A8 01 01 -> 192.168.1.1"""
    try:
        value = int(hex_text, 16)
    except ValueError:
        return None
    return ".".join(str((value >> shift) & 0xFF) for shift in (0, 8, 16, 24))


def parse_linux_route(text: str) -> list[dict]:
    """Default routes (destination 0.0.0.0 with the gateway flag) from /proc/net/route."""
    gateways = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        iface, destination, gateway, flags, metric = parts[0], parts[1], parts[2], parts[3], parts[6]
        try:
            if destination != "00000000" or not int(flags, 16) & RTF_GATEWAY:
                continue
            gateways.append({"ip": _hex_le_to_ip(gateway), "interface": iface,
                             "metric": int(metric)})
        except ValueError:
            continue
    return sorted((g for g in gateways if g["ip"]), key=lambda g: g["metric"])


WIN_INTERFACE = re.compile(rf"^\S.*?({IP})\s+-+\s+(0x[0-9a-fA-F]+)")
WIN_ENTRY = re.compile(rf"^\s+({IP})\s+({MAC_ANY})\s+(\S+)")


def parse_windows_arp(text: str) -> list[dict]:
    """`arp -a`:
    Interface: 192.168.1.10 --- 0xb
      Internet Address      Physical Address      Type
      192.168.1.1           a4-2b-b0-11-22-33     dynamic
    """
    entries, interface = [], None
    for line in text.splitlines():
        header = WIN_INTERFACE.match(line)
        if header:
            interface = header.group(1)          # our own IP on that interface
            continue
        entry = WIN_ENTRY.match(line)
        if entry:
            kind = entry.group(3).lower()
            entries.append({
                "ip": entry.group(1),
                "mac": entry.group(2).lower().replace("-", ":"),
                "interface": interface or "?",
                "complete": True,
                # "dynamic"/"dynamisch"/"dynamique" all start with "dyn"
                "type": "dynamic" if kind.startswith("dyn") else "static",
            })
    return entries


WIN_DEFAULT_ROUTE = re.compile(rf"^\s*0\.0\.0\.0\s+0\.0\.0\.0\s+({IP})\s+({IP})\s+(\d+)")


def parse_windows_route(text: str) -> list[dict]:
    """Default routes from `route print -4` ("On-link" lines are skipped
    automatically because "On-link" isn't an IP address)."""
    gateways = []
    for line in text.splitlines():
        match = WIN_DEFAULT_ROUTE.match(line)
        if match:
            gateways.append({"ip": match.group(1), "interface": match.group(2),
                             "metric": int(match.group(3))})
    return sorted(gateways, key=lambda g: g["metric"])


def collect(ctx):
    if ctx.platform == "linux":
        arp_raw = ctx.read_file("/proc/net/arp")
        route_raw = ctx.read_file("/proc/net/route")
        entries, gateways = parse_linux_arp(arp_raw), parse_linux_route(route_raw)
        source = "/proc/net/arp and /proc/net/route"
    else:
        arp_raw = ctx.run(["arp", "-a"])
        route_raw = ctx.run(["route", "print", "-4"])
        entries, gateways = parse_windows_arp(arp_raw), parse_windows_route(route_raw)
        source = "`arp -a` and `route print -4`"

    # The raw table is kept as the text until the enricher replaces it
    # with a clean one; it's never empty, so a run without enricher still works.
    return {
        "text": arp_raw if arp_raw.strip() else "(the ARP table is empty)",
        "records": entries,
        "meta": {"source": source, "platform": ctx.platform, "gateways": gateways,
                 "entries_read": len(entries)},
    }
