# Future Integration: strategy-transfer

## Current State
Tests whether ternary strategies transfer across domains. Key finding: transfer is **neutral** — 33% positive, 33% negative, 33% neutral. Strategies are domain-bound, not universal. Each domain requires its own training investment.

## Integration Opportunities

### With cross-room strategy sharing
strategy-transfer's neutral finding has a crucial implication for the fleet: rooms cannot share strategies directly. Each room must train its own strategies for its own domain. However, the training *methodology* transfers: how to select, mutate, and evaluate strategies is universal. strategy-transfer tells us what NOT to share (strategies) and what TO share (methods).

### With room-as-codespace
When an agent walks from Room A to Room B, it cannot carry Room A's strategies. But it CAN carry Room A's learning algorithm. The ensign pattern (load specialist on enter, unload on exit) aligns with this: each room's ensign trains strategies specific to that room's domain.

### With strategy-ecology
The 5 strategy species appear in every domain (Explorer, Diplomat, etc.), but their specific parameterizations are domain-bound. The species taxonomy transfers; the strategy vectors don't. strategy-transfer quantifies exactly how much transfers and what doesn't.

## Dormant Ideas Now Unlockable
The neutral finding was a research result. Now it's an architectural principle: rooms are domain-isolated. The fleet's architecture respects domain boundaries because strategy-transfer proved that crossing them doesn't help.

## Potential in Mature Systems
The fleet operates under strategy-transfer's constraint: each room trains independently, using shared methodology but not shared strategies. This is actually a feature, not a limitation — it ensures domain specificity and prevents cross-domain contamination.

## Cross-Pollination Ideas
- **evolution-ternary**: Evolution is room-local, not fleet-wide
- **strategy-ecology**: Species taxonomy transfers, parameters don't
- **room-as-codespace**: Room isolation is enforced by transfer properties

## Dependencies for Next Steps
- Document domain-isolation as an architectural principle
- Share learning methodology between rooms (not strategies)
- Measure transfer in real fleet rooms to validate the neutral finding
