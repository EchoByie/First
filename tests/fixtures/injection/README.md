Attack samples for tests/test_injection.py. Each line of attack_log.txt
is one known prompt-injection trick hidden in otherwise normal log data:
line 3 direct instruction, line 4 the same hidden with a zero-width
character, line 5 a fake SYSTEM role line, line 6 a forged answer, line 7
a fake end-of-data marker. Lines 2 and 8 are the real problem.
