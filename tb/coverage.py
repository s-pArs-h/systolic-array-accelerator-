"""Small functional-coverage collector.

Each bin is a named condition the tests must exercise at least once. The last
test in test_kmeans.py fails if any bin was never hit, so a passing run means
the stimulus actually reached every corner listed here.
"""


class Coverage:
    def __init__(self):
        self.bins = {}

    def define(self, name, goal=1):
        self.bins.setdefault(name, [0, goal])

    def hit(self, name, n=1):
        self.bins.setdefault(name, [0, 1])
        self.bins[name][0] += n

    def missing(self):
        return [k for k, (n, goal) in self.bins.items() if n < goal]

    def report(self):
        width = max(len(k) for k in self.bins)
        lines = ["Functional coverage:"]
        for name, (n, goal) in sorted(self.bins.items()):
            mark = "ok  " if n >= goal else "MISS"
            lines.append(f"  [{mark}] {name:<{width}}  hits={n} (goal {goal})")
        covered = sum(1 for n, goal in self.bins.values() if n >= goal)
        lines.append(f"  {covered}/{len(self.bins)} bins covered")
        return "\n".join(lines)


cov = Coverage()
