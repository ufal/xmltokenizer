"""Give elements their ids, byte-preserving.

The profile's `id_rules` say which elements get which ids (TEITOK: `w-N` tokens,
`p-N`, `s-N`, `l-N`, `e-N` for <lb>/<pb>/<gap>, ...). An id is added only where
the element has none (no attribute, an empty one, or the placeholder
`torenew`); an existing id is never changed, with one exception: a second
`<tok>` with an id already in use gets a new one (the rule's `fix_duplicates`).
Nothing else in the document moves: the only edits are inserted
` id="..."` attributes and replaced empty or duplicate id values.

Numbers follow the document position of the element within its rule (the
same way as TEITOK's xmlrenumber.pl): the third `<p>` wants `p-3`, and takes the
next free number when that one is in use. Rules with `after_max` (tokens)
continue after the highest existing `<prefix>-N` instead. A rule with
`per_token` numbers the children of a `<tok>`: `d-<n>-<k>`.

Ids that changed because they were duplicates are reported, not followed into
`@corresp` and the like: which of the two elements a reference meant is not
knowable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extract import (
    _build_scope_tag_matcher,
    _identify_scope_roots,
    _parse_events,
)

PLACEHOLDER = "torenew"


@dataclass
class RenumberReport:
    added: dict[str, int] = field(default_factory=dict)   # prefix -> count
    renumbered_duplicates: list[tuple[str, str]] = field(default_factory=list)  # old, new

    @property
    def total(self) -> int:
        return sum(self.added.values()) + len(self.renumbered_duplicates)

    def summary(self) -> str:
        parts = [f"{n} {p}-N" for p, n in sorted(self.added.items())]
        if self.renumbered_duplicates:
            parts.append(f"{len(self.renumbered_duplicates)} duplicate id(s) renumbered")
        return ", ".join(parts) if parts else "no ids to add"


def renumber(raw: bytes, profile: dict) -> tuple[bytes, RenumberReport]:
    """Return (`raw` with ids added, report). Raises ValueError if the profile
    has no `id_rules` or the document has no scope."""
    rules = profile.get("id_rules") or []
    if not rules:
        raise ValueError(
            f"profile {profile.get('name')!r} has no id_rules: nothing to number"
        )
    id_attr = (profile.get("tok_attrs") or {}).get("id_attr", "xml:id")
    by_tag: dict[str, int] = {}
    for i, rule in enumerate(rules):
        for tag in rule["elements"]:
            by_tag.setdefault(tag, i)

    events = _parse_events(raw)
    spans = _identify_scope_roots(events, _build_scope_tag_matcher(profile))
    if not spans:
        raise ValueError("no scope root matched: nothing to number")

    # every id in the document is taken, in or out of scope
    used: set[str] = set()
    for ev in events:
        if ev.kind == "start":
            for k, v in ev.data[1]:
                if k in ("id", "xml:id") and v:
                    used.add(v)

    # highest existing <prefix>-N of each after_max rule
    after: dict[int, int] = {}
    for i, rule in enumerate(rules):
        if rule.get("after_max"):
            pre = rule["prefix"] + "-"
            top = 0
            for u in used:
                if u.startswith(pre) and u[len(pre):].isdigit():
                    top = max(top, int(u[len(pre):]))
            after[i] = top

    report = RenumberReport()
    edits: list[tuple[int, int, bytes]] = []  # (start, end, replacement)
    position: dict[int, int] = {}             # rule -> elements seen
    next_after: dict[int, int] = dict(after)
    seen_ids: set[str] = set()
    # open elements: [tag, number of the token it is (or ""), per-rule child counters]
    stack: list[list] = []

    def new_id(rule_i: int, rule: dict, n: int) -> str:
        pre = rule["prefix"]
        if rule.get("after_max"):
            n = next_after[rule_i] + 1
        while f"{pre}-{n}" in used:
            n += 1
        if rule.get("after_max"):
            next_after[rule_i] = n
        return f"{pre}-{n}"

    def put(ev, value: str, has_attr: bool) -> None:
        span = _attr_value_span(raw, ev, id_attr) if has_attr else None
        if span is not None:
            edits.append((span[0], span[1], value.encode()))
        else:
            at = ev.byte_start + 1 + len(ev.data[0].encode("utf-8"))
            edits.append((at, at, f' {id_attr}="{value}"'.encode()))
        used.add(value)

    def count(prefix: str) -> None:
        report.added[prefix] = report.added.get(prefix, 0) + 1

    for s_idx, e_idx in spans:
        for i in range(s_idx, e_idx + 1):
            ev = events[i]
            if ev.kind == "end":
                if stack:
                    stack.pop()
                continue
            if ev.kind != "start":
                continue
            tag, attrs = ev.data
            has_attr = id_attr in dict(attrs)
            old = dict(attrs).get(id_attr, "")
            if old == PLACEHOLDER:
                old = ""
            tok_number = ""
            rule_i = by_tag.get(tag)
            if rule_i is not None:
                rule = rules[rule_i]
                parent = stack[-1] if stack else None
                if rule.get("per_token"):
                    if parent is not None and parent[1]:
                        parent[2][rule_i] = parent[2].get(rule_i, 0) + 1
                        if not old:
                            put(ev, f"{rule['prefix']}-{parent[1]}-{parent[2][rule_i]}", has_attr)
                            count(rule["prefix"])
                else:
                    position[rule_i] = position.get(rule_i, 0) + 1
                    duplicate = bool(old and rule.get("fix_duplicates") and old in seen_ids)
                    if not old or duplicate:
                        value = new_id(rule_i, rule, position[rule_i])
                        put(ev, value, has_attr)
                        if duplicate:
                            report.renumbered_duplicates.append((old, value))
                        else:
                            count(rule["prefix"])
                        old = value
                    if old:
                        seen_ids.add(old)
                    if rule.get("after_max") and old.startswith(rule["prefix"] + "-"):
                        tok_number = old[len(rule["prefix"]) + 1:]
            if not ev.empty:
                stack.append([tag, tok_number, {}])
    # apply from the end so earlier offsets stay valid
    out = bytearray(raw)
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        out[start:end] = repl
    return bytes(out), report


def id_counters(raw: bytes, profile: dict) -> tuple[list[int], list[int]]:
    """Start values for the token and sentence counters of a tokenization: the
    highest `w-N` / `s-N` (the profile's id templates) already used in `raw`, so
    new ids never repeat one the source has (TEITOK files whose `<l>` hold `s-N`)."""
    def prefix(key: str, default: str) -> str:
        tpl = (profile.get(key) or {}).get("id_template", default)
        return tpl.split("{n}")[0]

    wp, sp = prefix("tok_attrs", "w-{n}"), prefix("s_attrs", "s-{n}")
    tops = {wp: 0, sp: 0}
    for ev in _parse_events(raw):
        if ev.kind != "start":
            continue
        for k, v in ev.data[1]:
            if k in ("id", "xml:id"):
                for p in tops:
                    if v.startswith(p) and v[len(p):].isdigit():
                        tops[p] = max(tops[p], int(v[len(p):]))
    return [tops[wp]], [tops[sp]]


def _attr_value_span(raw: bytes, ev, name: str) -> tuple[int, int] | None:
    """Byte span of the value (inside the quotes) of attribute `name` in the
    start tag of `ev`, or None."""
    tag = raw[ev.byte_start:ev.byte_end]
    i = 1 + len(ev.data[0].encode("utf-8"))
    n = len(tag)
    target = name.encode("utf-8")
    while i < n:
        while i < n and tag[i:i + 1].isspace():
            i += 1
        if i >= n or tag[i:i + 1] in (b">", b"/"):
            return None
        j = i
        while j < n and tag[j:j + 1] not in (b"=", b" ", b"\t", b"\r", b"\n", b">", b"/"):
            j += 1
        key = tag[i:j]
        while j < n and tag[j:j + 1].isspace():
            j += 1
        if j >= n or tag[j:j + 1] != b"=":
            i = j + 1
            continue
        j += 1
        while j < n and tag[j:j + 1].isspace():
            j += 1
        quote = tag[j:j + 1]
        if quote not in (b'"', b"'"):
            return None
        end = tag.index(quote, j + 1)
        if key == target:
            return ev.byte_start + j + 1, ev.byte_start + end
        i = end + 1
    return None
