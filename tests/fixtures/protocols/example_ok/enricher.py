"""Test enricher: adds a line count computed by plain code."""


def enrich(data, ctx):
    data.text += f"\n[line count: {len(data.text.splitlines())}]"
    return data
