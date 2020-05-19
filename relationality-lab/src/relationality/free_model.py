"""Model of arbitrarily connected nodes."""

from typing_extensions import Literal

import numpy as np  # type: ignore
import numpy.random as nprand  # type: ignore

from relationality.util import Indexer
from relationality.fields import Histo, Distro, entropy, draw

ChoiceMode = Literal["round-robin", "uniform"]

class Model:
    """Model consisting of nodes with arbitrary connections.
        The underlying data for the community is a base_rank=1 distro field.
        There is one array dimension of field and at each point in this
        field is a distribution. The distribution at each point contains
        the probability of communicating with members.

        'round-robin' choice_mode steps through the members, cyclically.
        'uniform' choice_mode picks members randomly for each step.
    """
    community: Distro
    members: int
    update_coeff: float
    rr_index: int

    def __init__(
        self,
        members: int,
        update_coefficient: float = 0.1,
        choice_mode: ChoiceMode = "uniform",
    ):
        self.members = members
        self.community = Distro.uniform((members, members), base_rank=1)
        self.update_coeff = update_coefficient
        self.choice_mode = choice_mode

    def entropies(self) -> np.ndarray:
        """Calculate entropies of nodes."""
        return entropy(self.community)

    def communicate(self, member: Indexer) -> Indexer:
        """Send message from given index to drawn target."""

        # Get the distribution for the given member
        fiber = self.community.point(member)

        # Draw randomly with this distro the target member
        target = draw(fiber)

        # Define update envelope
        bump = np.exp(
            Histo.dirac(fiber.shape, target, self.update_coeff)
        )

        # Update distribution
        self.community.update_at(bump, member)

        # Return chosen target index
        return target

    def choose(self):
        """Choose the next member."""
        if self.choice_mode == "uniform":
            return nprand.randint(self.members)

        assert self.choice_mode == "round-robin"
        chosen = self.rr_index
        self.rr_index = (self.rr_index + 1) % self.members
        return chosen

    def step(self):
        """Step through simulation."""
        member = self.choose()
        self.communicate(member)
