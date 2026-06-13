from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel


class BaseAppraisalModel(nn.Module):
    """
    Shared functionality for every appraisal architecture.
    """

    def __init__(
        self,
        model_name: str,
        target_dims: list[str],
        dropout_p: float,
    ):
        super().__init__()

        self.target_dims = list(target_dims)
        self.num_labels = len(self.target_dims)

        self.base_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size

        self.dropout = nn.Dropout(dropout_p)

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Average only real tokens, excluding padding tokens.
        """

        expanded_mask = attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        summed_hidden = (
            hidden_states * expanded_mask
        ).sum(dim=1)

        token_counts = expanded_mask.sum(dim=1).clamp(min=1.0)

        return summed_hidden / token_counts

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = self.masked_mean_pool(
            outputs.last_hidden_state,
            attention_mask,
        )

        return self.dropout(pooled)

    def head_parameters(self):
        """
        Parameters excluding the pretrained transformer.
        """

        for name, parameter in self.named_parameters():
            if not name.startswith("base_model."):
                yield parameter


class FlatLinearAppraisalModel(BaseAppraisalModel):
    """
    Baseline 1:
        RoBERTa -> masked mean pooling -> Linear -> 21
    """

    def __init__(
        self,
        model_name: str,
        target_dims: list[str],
        dropout_p: float = 0.2,
    ):
        super().__init__(
            model_name=model_name,
            target_dims=target_dims,
            dropout_p=dropout_p,
        )

        self.output_layer = nn.Linear(
            self.hidden_size,
            self.num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        logits = self.output_layer(pooled)

        return torch.sigmoid(logits)


class FlatMLPAppraisalModel(BaseAppraisalModel):
    """
    Baseline 2:
        RoBERTa -> masked mean pooling -> shared MLP -> 21

    This is the capacity-control baseline for the grouped model.
    """

    def __init__(
        self,
        model_name: str,
        target_dims: list[str],
        shared_dim: int = 256,
        dropout_p: float = 0.2,
    ):
        super().__init__(
            model_name=model_name,
            target_dims=target_dims,
            dropout_p=dropout_p,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(shared_dim, self.num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        logits = self.head(pooled)

        return torch.sigmoid(logits)


class GroupedParallelAppraisalModel(BaseAppraisalModel):
    """
    Theory-informed model 1:

        RoBERTa
            -> shared projection
            -> relevance head
            -> implication head
            -> coping head
            -> normative head

    All four heads operate in parallel.
    """

    def __init__(
        self,
        model_name: str,
        target_dims: list[str],
        objective_groups: dict[str, list[str]],
        shared_dim: int = 256,
        group_hidden_dim: int = 128,
        dropout_p: float = 0.2,
    ):
        super().__init__(
            model_name=model_name,
            target_dims=target_dims,
            dropout_p=dropout_p,
        )

        self.objective_groups = OrderedDict(objective_groups)

        self.shared_projection = nn.Sequential(
            nn.Linear(self.hidden_size, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        self.group_heads = nn.ModuleDict({
            group_name: nn.Sequential(
                nn.Linear(shared_dim, group_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(
                    group_hidden_dim,
                    len(group_dims),
                ),
            )
            for group_name, group_dims
            in self.objective_groups.items()
        })

        self.dimension_location = self._build_dimension_location()

    def _build_dimension_location(self) -> dict[str, tuple[str, int]]:
        location = {}

        for group_name, group_dims in self.objective_groups.items():
            for local_idx, dim_name in enumerate(group_dims):
                location[dim_name] = (group_name, local_idx)

        return location

    def _flatten_group_outputs(
        self,
        group_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Restore the exact TARGET_DIMS order.
        """

        ordered_outputs = []

        for dim_name in self.target_dims:
            group_name, local_idx = self.dimension_location[dim_name]

            ordered_outputs.append(
                group_outputs[group_name][:, local_idx]
            )

        return torch.stack(ordered_outputs, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        shared = self.shared_projection(pooled)

        group_outputs = {
            group_name: head(shared)
            for group_name, head in self.group_heads.items()
        }

        logits = self._flatten_group_outputs(group_outputs)

        return torch.sigmoid(logits)


class GroupedSequentialAppraisalModel(BaseAppraisalModel):
    """
    Theory-informed model 2:

        shared text representation
              |
          relevance
              |
         implication
              |
            coping
              |
          normative

    Each later group is conditioned on the preceding group's
    hidden representation.
    """

    GROUP_ORDER = [
        "relevance",
        "implication",
        "coping",
        "normative",
    ]

    def __init__(
        self,
        model_name: str,
        target_dims: list[str],
        objective_groups: dict[str, list[str]],
        shared_dim: int = 256,
        group_hidden_dim: int = 128,
        dropout_p: float = 0.2,
    ):
        super().__init__(
            model_name=model_name,
            target_dims=target_dims,
            dropout_p=dropout_p,
        )

        self.objective_groups = OrderedDict(objective_groups)

        for group_name in self.GROUP_ORDER:
            if group_name not in self.objective_groups:
                raise ValueError(
                    f"Missing required sequential group: {group_name}"
                )

        self.shared_projection = nn.Sequential(
            nn.Linear(self.hidden_size, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

        self.relevance_layer = self._make_group_layer(
            input_dim=shared_dim,
            hidden_dim=group_hidden_dim,
            dropout_p=dropout_p,
        )

        conditioned_input_dim = shared_dim + group_hidden_dim

        self.implication_layer = self._make_group_layer(
            input_dim=conditioned_input_dim,
            hidden_dim=group_hidden_dim,
            dropout_p=dropout_p,
        )

        self.coping_layer = self._make_group_layer(
            input_dim=conditioned_input_dim,
            hidden_dim=group_hidden_dim,
            dropout_p=dropout_p,
        )

        self.normative_layer = self._make_group_layer(
            input_dim=conditioned_input_dim,
            hidden_dim=group_hidden_dim,
            dropout_p=dropout_p,
        )

        self.output_heads = nn.ModuleDict({
            group_name: nn.Linear(
                group_hidden_dim,
                len(self.objective_groups[group_name]),
            )
            for group_name in self.GROUP_ORDER
        })

        self.dimension_location = self._build_dimension_location()

    @staticmethod
    def _make_group_layer(
        input_dim: int,
        hidden_dim: int,
        dropout_p: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
        )

    def _build_dimension_location(self) -> dict[str, tuple[str, int]]:
        location = {}

        for group_name, group_dims in self.objective_groups.items():
            for local_idx, dim_name in enumerate(group_dims):
                location[dim_name] = (group_name, local_idx)

        return location

    def _flatten_group_outputs(
        self,
        group_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ordered_outputs = []

        for dim_name in self.target_dims:
            group_name, local_idx = self.dimension_location[dim_name]

            ordered_outputs.append(
                group_outputs[group_name][:, local_idx]
            )

        return torch.stack(ordered_outputs, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.encode(input_ids, attention_mask)
        shared = self.shared_projection(pooled)

        relevance_hidden = self.relevance_layer(shared)

        implication_hidden = self.implication_layer(
            torch.cat(
                [shared, relevance_hidden],
                dim=-1,
            )
        )

        coping_hidden = self.coping_layer(
            torch.cat(
                [shared, implication_hidden],
                dim=-1,
            )
        )

        normative_hidden = self.normative_layer(
            torch.cat(
                [shared, coping_hidden],
                dim=-1,
            )
        )

        hidden_by_group = {
            "relevance": relevance_hidden,
            "implication": implication_hidden,
            "coping": coping_hidden,
            "normative": normative_hidden,
        }

        group_outputs = {
            group_name: self.output_heads[group_name](
                hidden_by_group[group_name]
            )
            for group_name in self.GROUP_ORDER
        }

        logits = self._flatten_group_outputs(group_outputs)

        return torch.sigmoid(logits)


def build_model(
    model_type: str,
    model_name: str,
    target_dims: list[str],
    objective_groups: dict[str, list[str]],
    shared_dim: int = 256,
    group_hidden_dim: int = 128,
    dropout_p: float = 0.2,
) -> BaseAppraisalModel:
    """
    Factory function so train.py does not need architecture-specific code.
    """

    if model_type == "flat_linear":
        return FlatLinearAppraisalModel(
            model_name=model_name,
            target_dims=target_dims,
            dropout_p=dropout_p,
        )

    if model_type == "flat_mlp":
        return FlatMLPAppraisalModel(
            model_name=model_name,
            target_dims=target_dims,
            shared_dim=shared_dim,
            dropout_p=dropout_p,
        )

    if model_type == "grouped_parallel":
        return GroupedParallelAppraisalModel(
            model_name=model_name,
            target_dims=target_dims,
            objective_groups=objective_groups,
            shared_dim=shared_dim,
            group_hidden_dim=group_hidden_dim,
            dropout_p=dropout_p,
        )

    if model_type == "grouped_sequential":
        return GroupedSequentialAppraisalModel(
            model_name=model_name,
            target_dims=target_dims,
            objective_groups=objective_groups,
            shared_dim=shared_dim,
            group_hidden_dim=group_hidden_dim,
            dropout_p=dropout_p,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def freeze_encoder(model: BaseAppraisalModel) -> None:
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False


def unfreeze_encoder(model: BaseAppraisalModel) -> None:
    for parameter in model.base_model.parameters():
        parameter.requires_grad = True


def print_trainable_parameters(model: nn.Module) -> None:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {trainable:,}/{total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.2f}%")