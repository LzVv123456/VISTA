from dataclasses import dataclass
from typing import Optional
import torch
from transformers import PreTrainedModel
from llm_layers import get_layers
from myutils import PCA


@dataclass
class ResidualStream:
    hidden: torch.Tensor


class ForwardTrace:
    def __init__(self):
        self.residual_stream: Optional[ResidualStream] = ResidualStream(
            hidden=[],
        )
        self.attentions: Optional[torch.Tensor] = None


class ForwardTracer:
    def __init__(self, model: PreTrainedModel, forward_trace: ForwardTrace):
        self._model = model
        self._forward_trace = forward_trace
        self._layers = get_layers(model)
        self._hooks = []

    def __enter__(self):
        self._register_forward_hooks()


    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()

        if exc_type is None:
            residual_stream = self._forward_trace.residual_stream

            if residual_stream.hidden[0] == []:
                residual_stream.hidden.pop(0)

            for key in residual_stream.__dataclass_fields__.keys():
                acts = getattr(residual_stream, key)
                # TODO: this is a hack, fix it
                if key != "hidden" and not self._with_submodules:
                    continue

                nonempty_layer_acts = [layer_acts for layer_acts in acts if layer_acts != []][0]
                final_shape = torch.cat(nonempty_layer_acts, dim=0).shape
                for i, layer_acts in enumerate(acts):
                    if layer_acts == []:
                        acts[i] = torch.zeros(final_shape)
                    else:
                        acts[i] = torch.cat(layer_acts, dim=0)
                acts = torch.stack(acts).transpose(0, 1)
                setattr(residual_stream, key, acts)

    def _register_forward_hooks(self):
        model = self._model
        hooks = self._hooks

        residual_stream = self._forward_trace.residual_stream

        def store_activations(residual_stream: ResidualStream, acts_type: str, layer_num: int):
            def hook(model, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.float().to("cpu", non_blocking=True)

                acts = getattr(residual_stream, acts_type)
                while len(acts) < layer_num + 1:
                    acts.append([])
                try:
                    acts[layer_num].append(out)
                except IndexError:
                    print(len(acts), layer_num)  
            return hook

        # Do not include the embedding layer for MLLMs
        # embedding_hook = get_embedding_layer(self._model).register_forward_hook(
        #     store_activations(residual_stream, "hidden", 0)
        # )
        # hooks.append(embedding_hook)

        for i, layer in enumerate(self._layers):
            hidden_states_hook = layer.register_forward_hook(store_activations(residual_stream, "hidden", i + 1))
            hooks.append(hidden_states_hook)


def get_hiddenstates(model, kwargs_list):
    h_all = []
    for example_id in range(len(kwargs_list)):
        embeddings_for_all_styles= []
        for style_id in range(len(kwargs_list[example_id])):
            forward_trace = ForwardTrace()
            context_manager = ForwardTracer(model, forward_trace)
            with context_manager:
                _ = model(
                use_cache=True,
                **kwargs_list[example_id][style_id],
                )
                h = forward_trace.residual_stream.hidden
            embedding_token = []
            for layer in range(len(h)):
                embedding_token.append(h[layer][:,-1])
            embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
            embeddings_for_all_styles.append(embedding_token)
        h_all.append(tuple(embeddings_for_all_styles))
    return h_all


def obtain_vsv(args, model, kwargs_list, rank=1):
    hidden_states = get_hiddenstates(model, kwargs_list) # each element, layer x len_tokens x dim
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)
    neg_emb = torch.stack(neg_all).mean(0)
    pos_emb = torch.stack(pos_all).mean(0)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=0,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, (neg_emb).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))


def add_logits_flag(model, args):
    assert not hasattr(model, 'logits_aug')
    assert not hasattr(model, 'logits_layers')
    assert not hasattr(model, 'logits_alpha')
    model.logits_aug = args.logits_aug
    model.logits_layers = args.logits_layers
    model.logits_alpha = args.logits_alpha


def remove_logits_flag(model):
    del model.logits_aug
    del model.logits_layers
    del model.logits_alpha