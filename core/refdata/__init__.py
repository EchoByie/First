"""Local reference data (offline lookup tables) kept in one SQLite file.

The first dataset is "oui": the IEEE registries that map the start of a MAC
address to the company it was assigned to. The design leaves room for more
datasets later; each has its own table plus rows in the shared `sources`
table recording where the data came from and when.
"""
