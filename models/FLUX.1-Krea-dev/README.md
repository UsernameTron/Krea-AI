---
language:
- en
license: other
license_name: flux-1-dev-non-commercial-license
license_link: LICENSE.md
extra_gated_prompt: >-
  By clicking "Agree", you agree to the [FluxDev Non-Commercial License
  Agreement](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
  and acknowledge the [Acceptable Use
  Policy](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/POLICY.md).
tags:
- text-to-image
- image-generation
- flux
- diffusers
pipeline_tag: text-to-image
base_model:
- black-forest-labs/FLUX.1-dev
library_name: diffusers
---

![FLUX.1 Krea [dev] Grid](./teaser.png)


`FLUX.1 Krea [dev]` is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions.
For more information, please read our [blog post](https://bfl.ai/announcements/flux-1-krea-dev) and [Krea's blog post](https://www.krea.ai/blog/flux-krea-open-source-release).


# Key Features
1. Cutting-edge output quality, with a focus on aesthetic photography.
2. Competitive prompt following, matching the performance of closed source alternatives.
3. Trained using guidance distillation, making `FLUX.1 Krea [dev]` more efficient.
4. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
5. Generated outputs can be used for personal, scientific, and commercial purposes, as described in the [flux-1-dev-non-commercial-license](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

# Usage
`FLUX.1 Krea [dev]` can be used as a drop-in replacement in every system that supports the original `FLUX.1 [dev]`.
A reference implementation of `FLUX.1 [dev]` is in our dedicated [github repository](https://github.com/black-forest-labs/flux).
Developers and creatives looking to build on top of `FLUX.1 [dev]` are encouraged to use this as a starting point.

`FLUX.1 Krea [dev]` is also available in both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [diffusers](https://github.com/huggingface/diffusers).

## ComfyUI
To use `FLUX.1 Krea [dev]` in [Comfy UI](https://github.com/comfyanonymous/ComfyUI) download the `*.safetensors` weights [here](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/resolve/main/flux1-krea-dev.safetensors)

## 🧨 diffusers
To use `FLUX.1 Krea [dev]` in [diffusers](https://github.com/huggingface/diffusers), first install or upgrade diffusers

```shell
pip install -U diffusers
```

Then you can use `FluxPipeline` to run the model

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU VRAM

prompt = "A frog holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]
image.save("flux-krea-dev.png")
```

To learn more check out the [diffusers](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) documentation

---

# Limitations
- This model is not intended or able to provide factual information.
- As a statistical model this checkpoint might amplify existing societal biases.
- The model may fail to generate output that matches the prompts.
- Prompt following is heavily influenced by the prompting-style.

---

# Out-of-Scope Use
The model and its derivatives may not be used

- In any way that violates any applicable national, federal, state, local or international law or regulation.
- For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; including but not limited to the solicitation, creation, acquisition, or dissemination of child exploitative content.
- To generate or disseminate verifiably false information and/or content with the purpose of harming others.
- To generate or disseminate personal identifiable information that can be used to harm an individual.
- To harass, abuse, threaten, stalk, or bully individuals or groups of individuals.
- To create non-consensual nudity or illegal pornographic content.
- For fully automated decision making that adversely impacts an individual's legal rights or otherwise creates or modifies a binding, enforceable obligation.
- Generating or facilitating large-scale disinformation campaigns.
- Please reference our [content filters](https://github.com/black-forest-labs/flux/blob/main/src/flux/content_filters.py) to avoid such generations.

---

# Risks 

Black Forest Labs (BFL) and Krea are committed to the responsible development of generative AI technology. Prior to releasing FLUX.1 Krea [dev], BFL and Krea collaboratively evaluated and mitigated a number of risks in the FLUX.1 Krea [dev]  model and services, including the generation of unlawful content. We implemented a series of pre-release mitigations to help prevent misuse by third parties, with additional post-release mitigations to help address residual risks:
1. **Pre-training mitigation.** BFL filtered pre-training data for multiple categories of “not safe for work” (NSFW) and unlawful content to help prevent a user generating unlawful content in response to text prompts or uploaded images.
2. **Post-training mitigation.** BFL has partnered with the Internet Watch Foundation, an independent nonprofit organization dedicated to preventing online abuse, to filter known child sexual abuse material (CSAM) from post-training data. Subsequently, BFL and Krea undertook multiple rounds of targeted fine-tuning to provide additional mitigation against potential abuse. By inhibiting certain behaviors and concepts in the trained model, these techniques can help to prevent a user generating synthetic CSAM or nonconsensual intimate imagery (NCII) from a text prompt.
3. **Pre-release evaluation.** Throughout this process, BFL conducted internal and external third-party evaluations of model checkpoints to identify further opportunities for improvement. The third-party evaluations focused on eliciting CSAM and NCII through adversarial testing of the text-to-image model with text-only prompts. We also conducted internal evaluations of the proposed release checkpoints, comparing the model with other leading openly-available generative image models from other companies. The final FLUX.1 Krea [dev] open-weight model checkpoint demonstrated very high resilience against violative inputs, demonstrating higher resilience than other similar open-weight models across these risk categories.  Based on these findings, we approved the release of the FLUX.1 Krea [dev] model as openly-available weights under a non-commercial license to support third-party research and development.
4. **Inference filters.** The BFL Github repository for the open FLUX.1 Krea [dev] model includes filters for illegal or infringing content. Filters or manual review must be used with the model under the terms of the FLUX.1 [dev] Non-Commercial License. We may approach known deployers of the FLUX.1 Krea [dev] model at random to verify that filters or manual review processes are in place.
5. **Policies.** Our FLUX.1 [dev] Non-Commercial License prohibits the generation of unlawful content or the use of generated content for unlawful, defamatory, or abusive purposes. Developers and users must consent to these conditions to access the FLUX.1 Krea [dev] model.
6. **Monitoring.** BFL is monitoring for patterns of violative use after release, and may ban developers who we detect intentionally and repeatedly violate our policies. Additionally, BFL provides a dedicated email address (safety@blackforestlabs.ai) to solicit feedback from the community. BFL maintains a reporting relationship with organizations such as the Internet Watch Foundation and the National Center for Missing and Exploited Children, and BFL welcomes ongoing engagement with authorities, developers, and researchers to share intelligence about emerging risks and develop effective mitigations.

---

# License
This model falls under the [`FLUX.1 [dev]` Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).