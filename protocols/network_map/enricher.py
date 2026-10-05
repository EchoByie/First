"""Network map enricher: clean the ARP entries and add facts, all in plain code.

For every entry:
  1. DROP what isn't a real neighbour device:
       - incomplete entries (the OS asked, nobody answered: MAC 00:00:...)
       - broadcast       (MAC ff:ff:ff:ff:ff:ff, IP 255.255.255.255)
       - multicast       (IP 224.0.0.0 - 239.255.255.255, or a multicast MAC:
                          lowest bit of the first byte set, e.g. 01:00:5e:...)
  2. LOOK UP the vendor in the local IEEE database (offline).
  3. FLAG a randomized MAC: the "locally administered" bit (second-lowest
     bit of the first byte) is set, so the address was made up by the
     device (phones do this for privacy) or by software (VMs, containers),
     not burned in at the factory. Such MACs have no real vendor.
  4. FLAG the gateway (the default route's next hop: usually your router).
  5. NOTE oddities: one MAC answering for several IPs, or one IP seen
     with several MACs. Often harmless (a router with many addresses),
     but the second can also be a sign of ARP spoofing.

The model then sees one line per device, e.g.

  device 3: ip 192.168.1.20 | mac 00:1b:a9:12:34:56 | vendor Example Printer Works (MA-L)
            | randomized no | gateway no | interface wlan0 | entry dynamic

Each piece is short and exact, so the model can quote it as evidence.
"""

from __future__ import annotations

import ipaddress
from collections import defaultdict

from core.refdata.db import normalise_mac

BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"
INCOMPLETE_MAC = "00:00:00:00:00:00"

# Small, well-known hints for locally administered prefixes. Only used as
# a hint in the facts; never as proof.
LOCAL_PREFIX_HINTS = {
    "02:42": "commonly used by Docker containers",
}


def first_byte(mac: str) -> int:
    return int(mac[:2], 16)


def is_multicast_mac(mac: str) -> bool:
    return bool(first_byte(mac) & 0x01)


def is_locally_administered(mac: str) -> bool:
    return bool(first_byte(mac) & 0x02)


def drop_reason(entry: dict) -> str | None:
    mac, ip = entry["mac"], entry["ip"]
    if not entry.get("complete", True) or mac == INCOMPLETE_MAC:
        return "incomplete"
    if normalise_mac(mac) is None:
        return "invalid MAC"
    if mac == BROADCAST_MAC or ip == "255.255.255.255":
        return "broadcast"
    try:
        if ipaddress.ip_address(ip).is_multicast:
            return "multicast"
    except ValueError:
        return "invalid IP"
    if is_multicast_mac(mac):
        return "multicast"
    return None


def describe_vendor(mac: str, db) -> tuple[str, str]:
    """Return (vendor text, randomized yes/no/...)."""
    if is_locally_administered(mac):
        cid = db.vendor(mac, include_cid=True) if db else None
        if cid and cid.registry == "CID":
            return f"{cid.organization} (CID)", "no (locally administered, registered CID)"
        hint = LOCAL_PREFIX_HINTS.get(mac[:5])
        return ("none (randomized MAC)",
                "yes" + (f" ({mac[:5]} prefix {hint})" if hint else ""))
    if db is None:
        return "unknown (no reference data imported)", "no"
    match = db.vendor(mac)
    if match is None:
        return "not in IEEE registry", "no"
    return f"{match.organization} ({match.registry})", "no"


def enrich(data, ctx):
    gateways = data.meta.get("gateways", [])
    gateway_ips = {g["ip"] for g in gateways}

    kept, dropped = [], defaultdict(int)
    for entry in data.records:
        reason = drop_reason(entry)
        if reason:
            dropped[reason] += 1
        else:
            kept.append(dict(entry))

    db = ctx.open_refdata()
    try:
        sources = db.sources("oui") if db else []
        for entry in kept:
            entry["vendor"], entry["randomized"] = describe_vendor(entry["mac"], db)
            entry["gateway"] = entry["ip"] in gateway_ips
    finally:
        if db:
            db.close()

    # Oddities, worked out by code
    ips_per_mac, macs_per_ip = defaultdict(set), defaultdict(set)
    for e in kept:
        ips_per_mac[e["mac"]].add(e["ip"])
        macs_per_ip[e["ip"]].add(e["mac"])

    kept.sort(key=lambda e: tuple(int(p) for p in e["ip"].split(".")))
    lines = []
    for n, e in enumerate(kept, start=1):
        lines.append(
            f"device {n}: ip {e['ip']} | mac {e['mac']} | vendor {e['vendor']} "
            f"| randomized {e['randomized']} | gateway {'yes' if e['gateway'] else 'no'} "
            f"| interface {e['interface']} | entry {e['type']}"
        )

    facts = [
        f"source: {data.meta.get('source', '?')} (read only; nothing was pinged or scanned)",
        f"entries read: {data.meta.get('entries_read', len(data.records))}",
        "entries dropped: " + (", ".join(f"{n} {r}" for r, n in sorted(dropped.items()))
                               if dropped else "none"),
        f"devices listed: {len(kept)}",
    ]
    if gateways:
        facts.append("default gateway: " + ", ".join(
            f"{g['ip']} via {g['interface']} (metric {g['metric']})" for g in gateways))
    else:
        facts.append("default gateway: none found in the routing table")
    for g in gateway_ips:
        if g not in macs_per_ip:
            facts.append(f"gateway {g} is not in the ARP table right now")
    randomized = sum(e["randomized"].startswith("yes") for e in kept)
    facts.append(f"randomized MACs: {randomized} of {len(kept)}")
    for mac, ips in sorted(ips_per_mac.items()):
        if len(ips) > 1:
            facts.append(f"one MAC on several IPs: {mac} answers for {', '.join(sorted(ips))}")
    for ip, macs in sorted(macs_per_ip.items()):
        if len(macs) > 1:
            facts.append(f"one IP with several MACs: {ip} seen with {', '.join(sorted(macs))}")
    if sources:
        newest = min(s.age_days() for s in sources)
        facts.append(f"vendor data: IEEE registries {', '.join(s.registry for s in sources)}, "
                     f"imported {newest:.0f} days ago")
    else:
        facts.append("vendor data: not available (run: aicore refdata import ...)")
        data.caveats.append("No MAC vendor data was available, so vendors are unknown. "
                            "Import the IEEE registries to improve the guesses.")

    data.preamble = "=== NETWORK FACTS COMPUTED BY CODE ===\n" + "\n".join(facts) + \
                    "\n=== DEVICES (one line each) ==="
    data.text = "\n".join(lines) if lines else "(no devices left after filtering)"
    data.records = kept
    data.meta["dropped"] = dict(dropped)
    return data
