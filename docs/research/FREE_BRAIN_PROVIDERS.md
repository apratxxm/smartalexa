# Free hosted brain providers

**Read date: 2026-10-24 (India).** Every linked fact below was checked on that date; source dates below mean date read. Official sources only. No signups, keys, inference requests, or latency measurements were performed. “Unverified” means no adequate public confirmation or no account/runtime evidence; it does not mean unavailable.

Workload arithmetic: 50–150 requests/day × (1,500–3,000 input + 150 output) = **82,500–472,500 tokens/day**, before synthesis, retries, or additional reasoning tokens. Tool definitions/history must fit inside the input estimate. An occasional synthesis is another request. None of these free plans establishes an India-specific ≤1 s time-to-first-token (TTFT) guarantee.

## Comparison

RPM/RPD = requests/minute/day; TPM/TPD = tokens/minute/day. NP = not publicly specified in the sources inspected; **unverified**, not unlimited. Model lists are relevant examples rather than an exhaustive catalog. Account dashboards may impose different limits.

| Provider / recurring $0 status | Models with function/tool calling | RPM; RPD; TPM; TPD | Card / billing requirement | Free prompt training / human access | India availability | Latency evidence / workload fit |
| --- | --- | --- | --- | --- | --- | --- |
| **Cloudflare Workers AI — yes**, Workers Free allocation [C1] | `@cf/qwen/qwen3-30b-a3b-fp8`; `@cf/openai/gpt-oss-20b` both explicitly support function calling [C2] [C3] | **300; NP; NP; token-equivalent varies**. Hard shared allocation **10,000 neurons/day**, reset 00:00 UTC; Free stops at quota [C1] [C4] | Free account; no card needed for signup [C5]. Selected models are outside the paid-only list [C1]; no Workers Paid plan or prepaid credits required | **No training**; ordinary stateless inference has no retained input/output after processing. Limited legal/security exceptions; absence of all human access is **unverified** [C6] | India-specific signup/API eligibility **unverified**; no India allowlist confirmation found | GPT-OSS 20B described as lower latency, qualitatively [C3]. India TTFT **unverified**. Qwen estimated **575–2,767 neurons/day**; GPT-OSS 20B **1,568–8,796**, excluding extra reasoning/synthesis [C1] |
| **Groq — yes**, Free plan [G1] [G2] | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`; tools documented. GPT-OSS lacks parallel tool calls [G3] | Published table: **30; 1,000; 8,000; 200,000 per model** [G1]. Exact Free account limits **unverified**; see documentation caveat below | Payment method required to upgrade to Developer, not to remain Free [G2]; Free excess returns 429 rather than a charge [G7] | Training: service-only use inferred from agreement §8.2; blanket no-training prohibition **unverified**. Reliability/abuse logs can retain content up to 30 days; all customers can enable ZDR. Authorized service personnel access is possible [G4] [G5] | Official billing FAQ explicitly covers India customers [G2]; Free signup from India **not tested** | Published generation speed **1,000 tokens/s (20B), 500 (120B), 450 (Qwen)**, not TTFT [G6]. One 200k-token allowance supports only **63–121 estimated turns/day**, so cannot alone cover the upper case [G1] |
| **Google Gemini API / AI Studio — yes**, selected free models [A1] | `gemini-3.5-flash-lite` explicitly supports function calling [A2]; free input/output confirmed [A1] | **RPM/RPD/TPM/TPD: account/model-specific, all numeric values unverified.** Public rate page sends users to AI Studio; limits are per project [A3] | Free tier requires no linked billing account; linking billing upgrades the project [A3] [A4]. Use a project without active billing | **Yes**, unpaid inputs/outputs may improve products; **human reviewers may read/annotate them**. Terms say not to submit sensitive, confidential, or personal information [A5] | **India explicitly listed**, subject to eligibility requirements [A6] | Model card describes speed for latency-sensitive tasks; no numerical India TTFT verified [A7]. **50–150/day capacity unverified until account quotas are visible** |
| **Mistral Studio / La Plateforme — yes**, default Free mode, evaluation/prototyping [M1] [M2] | Official Free quickstart uses `mistral-small-latest` [M1]. `ministral-8b-2512` supports function calling [M3], but **specific free model entitlement is unverified** | **RPM/RPD/TPM/TPD numeric values unverified**. Official controls are RPS, TPM, **tokens/month**; limits are per organization/model and shown in Admin → Limits [M2] | **No credit card required** for Free API access [M1] | **Training allowed by default**, API opt-out available in Admin → Privacy → Anonymous improvement data [M4] [M5]. Free ZDR unavailable; human-review scope **unverified** [M6] | India-specific eligibility **unverified** | No usable numerical free-tier India TTFT found; **capacity and model eligibility unverified**, therefore a reserve candidate |
| **OpenRouter `:free` — yes**, but insufficient daily requests [O1] | `openai/gpt-oss-120b:free` is free and supports native function calling [O2]; `openrouter/free` filters for tools but chooses among models [O3] | **RPM unverified** (limits table values missing in fetched page); **50 RPD at $0** [O1] [O4]; TPM/TPD NP, upstream caps may apply | Free account requires no payment option [O1]. Buying credits to raise caps violates this task's $0 budget | OpenRouter retains prompts only by opt-in; **upstream providers may log/train**, controlled by routing/privacy settings. Human-review policy is **endpoint-specific/unverified** [O5] | India-specific account and selected upstream availability **unverified** | OpenRouter publishes provider TTFT metrics, but no numerical current free-endpoint value was retrievable [O6]. **50/day leaves no synthesis headroom even at minimum usage** |
| **Cerebras — no recurring free tier; exclude** [B1] [B2] | No qualifying free model. Trial includes `gpt-oss-120b`, `qwen-3.8-27b`; tool capability not audited for the excluded trial | Trial: **5 RPM; RPD NP; 30k uncached / 90k total TPM; 1M TPD**. These are **trial**, not renewable-free limits [B1] | **Verified payment method mandatory**, then $5 credits expiring after 30 days. No API/playground access without it [B1] | No qualifying tier; trial training/human policy **unverified** | No qualifying tier; India trial eligibility **unverified** | Excluded: a time/credit-bounded trial cannot sustain a $0 assistant. No India TTFT measured |
| **GitHub Models — retired; exclude** [H1] | None: playground, catalog, inference API, and BYOK retired **2026-07-30** | N/A: retired | No usable Models API key path | N/A: retired service | N/A: unavailable to all customers | N/A. GitHub Copilot is a separate service, not a replacement free brain API [H1] |

## Ranked shortlist for this assistant

1. **Cloudflare Workers AI / Qwen3 30B A3B:** strongest documented recurring-quota fit and explicit no-training policy. Arithmetic: `input/1M × 4,625 + output/1M × 30,475` neurons; even 150 × (3k + 150) is about 2,767/day. Actual semantic tool quality, extra reasoning consumption, India access, and ≤1 s TTFT remain **unverified**. GPT-OSS 20B is an alternative with much less quota margin at the upper workload [C1] [C2] [C3].
2. **Groq / GPT-OSS 20B:** strongest numerical published generation-speed signal; direct tool API and Free plan. **Good lower-usage candidate**, not proven sufficient for 150/day: 200k/(3,000+150) ≈ 63 turns, or ≈121 at 1,500+150. Do not assume cross-model quotas aggregate until the account console confirms it; cached tokens are excluded by Groq's rules but no cache hit rate is assumed [G1] [G3] [G6].
3. **Gemini / 3.5 Flash-Lite:** explicitly available in India, free token pricing, native function calling, and a latency-oriented model. **Conditional on the user's actual free project quota** supporting ≥150 requests/day plus synthesis and adequate TPM; free data terms allow training/human review. No public quota number or India TTFT was established [A1] [A2] [A3] [A5] [A6] [A7].

This ranks **candidates to qualify**, not three fully verified matches. None is proved to satisfy all throughput, privacy, and India-latency requirements. Mistral is below these because free model entitlements and numeric quotas remain account-dependent; OpenRouter fails the upper daily request requirement; Cerebras and GitHub Models are excluded.

Groq documentation caveat: the rate page exposes “Free Plan Limits” and “Developer Plan Limits,” but adjacent prose calls the displayed limits Developer base limits. The returned table has the values above; **the user's Free console is required to settle that inconsistency** [G1].

## User steps: account and key only

- **Cloudflare:** Sign up/sign in at [Cloudflare](https://dash.cloudflare.com/sign-up), retain **Workers Free**. Dashboard → Workers AI → **Use REST API** → **Create a Workers AI API Token** → review prefilled permissions → **Create API Token** → copy/store it. Copy **Account ID** on the same page. Custom tokens require Workers AI Read + Edit. No domain or Worker deployment is required by this REST key flow [C7].
- **Groq:** Sign up/sign in to [Groq Console](https://console.groq.com/), stay **Free**, open [API Keys](https://console.groq.com/keys), create/name a key, and copy/store it. Read **Limits** for the three candidate model IDs; do not activate Developer billing [G2] [G8]. Exact post-login key-form labels beyond the public quickstart are **unverified**.
- **Gemini:** Sign in with a Google account at [AI Studio API Keys](https://aistudio.google.com/api-keys), accept applicable terms yourself. A new user gets a default project/key; otherwise choose/create a project without billing, or import an existing **unbilled** project through Dashboard → Projects → Import projects. API Keys → **Create API key** → choose project → copy/store key. Check project's **Rate limits**; do not link Cloud Billing or select a paid key. New keys are authorization keys by default [A3] [A4] [A8].
- **Mistral:** Create/sign in to a [Mistral Studio account](https://console.mistral.ai/), remain **Free**. API Keys → **Create new key** → name + expiration + connector scope → **Create new key** → immediately copy/store it. Admin → Privacy → disable **Anonymous improvement data** if desired; this is separate from Vibe training settings. Inspect Admin → Limits for model entitlement and quotas [M1] [M2] [M5].
- **OpenRouter:** Create/sign in to an [OpenRouter account](https://openrouter.ai/), open [Keys](https://openrouter.ai/settings/keys) → create a named key → copy/store it; the official auth guide documents name and optional credit limit. Retain zero balance/no top-ups and use only explicitly free model IDs. Exact current post-login button labels are **unverified** [O1] [O7].
- **Cerebras / GitHub Models:** No compliant ongoing-free key setup to recommend: Cerebras requires payment verification for an expiring trial; GitHub Models is retired [B1] [H1].

<!-- Official source references: every URL read 2026-10-24. -->

[C1]: https://developers.cloudflare.com/workers-ai/platform/pricing/ "Workers AI pricing, free allocation, neuron conversion and paid-only exclusions; read 2026-10-24"
[C2]: https://developers.cloudflare.com/workers-ai/models/qwen3-30b-a3b-fp8/ "Qwen3 30B A3B capabilities; read 2026-10-24"
[C3]: https://developers.cloudflare.com/workers-ai/models/gpt-oss-20b/ "GPT-OSS 20B capabilities and qualitative latency; read 2026-10-24"
[C4]: https://developers.cloudflare.com/workers-ai/platform/limits/ "Workers AI rate limits; read 2026-10-24"
[C5]: https://developers.cloudflare.com/labs/workers "Cloudflare account signup without a card; read 2026-10-24"
[C6]: https://developers.cloudflare.com/workers-ai/platform/data-usage/ "Workers AI data use and exceptions; read 2026-10-24"
[C7]: https://developers.cloudflare.com/workers-ai/get-started/rest-api/ "Workers AI REST token/account setup; read 2026-10-24"
[G1]: https://console.groq.com/docs/rate-limits "Groq model quotas, cache counting and console authority; read 2026-10-24"
[G2]: https://console.groq.com/docs/billing-faqs "Groq billing FAQ, upgrade and India billing; read 2026-10-24"
[G3]: https://console.groq.com/docs/tool-use/overview "Groq tool/model capability matrix; read 2026-10-24"
[G4]: https://console.groq.com/docs/your-data "Groq retention and all-customer ZDR; read 2026-10-24"
[G5]: https://console.groq.com/docs/legal/services-agreement "Groq Services Agreement §§8.1–8.2 and 10.1; read 2026-10-24"
[G6]: https://console.groq.com/docs/models "Groq published model throughput; read 2026-10-24"
[G7]: https://community.groq.com/t/do-i-get-charged-for-anything-on-the-groq-free-plan/832 "Groq official Free-plan charging FAQ; read 2026-10-24"
[G8]: https://console.groq.com/docs/quickstart "Groq key quickstart; read 2026-10-24"
[A1]: https://ai.google.dev/gemini-api/docs/pricing "Gemini per-model free prices and data-use flags; read 2026-10-24"
[A2]: https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite "Gemini 3.5 Flash-Lite function calling; read 2026-10-24"
[A3]: https://ai.google.dev/gemini-api/docs/rate-limits "Gemini project quotas and Free versus billing tiers; read 2026-10-24"
[A4]: https://ai.google.dev/gemini-api/docs/billing "Gemini billing; read 2026-10-24"
[A5]: https://ai.google.dev/gemini-api/terms "Gemini unpaid-services training/human review terms; read 2026-10-24"
[A6]: https://ai.google.dev/gemini-api/docs/available-regions "Gemini available regions including India; read 2026-10-24"
[A7]: https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/ "Gemini 3.5 Flash-Lite model card, latency-sensitive positioning; read 2026-10-24"
[A8]: https://ai.google.dev/gemini-api/docs/api-key "AI Studio projects and key creation; read 2026-10-24"
[M1]: https://docs.mistral.ai/getting-started/quickstarts/studio/activate-and-generate-api-key "Mistral default Free API mode and key creation; read 2026-10-24"
[M2]: https://help.mistral.ai/en/articles/698531-why-am-i-hitting-api-rate-limits-and-how-do-i-increase-them "Mistral quota dimensions, plan scope and Admin limits; read 2026-10-24"
[M3]: https://docs.mistral.ai/models/ministral-3-8b-25-12 "Ministral 3 8B function calling; read 2026-10-24"
[M4]: https://help.mistral.ai/en/articles/347617-do-you-use-my-user-data-to-train-your-artificial-intelligence-models "Mistral Free-mode training; read 2026-10-24"
[M5]: https://help.mistral.ai/en/articles/455207-can-i-opt-out-of-my-input-or-output-data-being-used-for-training "Mistral API training opt-out; read 2026-10-24"
[M6]: https://help.mistral.ai/en/articles/347612-can-i-activate-zero-data-retention-zdr "Mistral ZDR restricted to pay-as-you-go; read 2026-10-24"
[O1]: https://openrouter.ai/pricing/ "OpenRouter Free pricing: 50 requests/day, no payment option; read 2026-10-24"
[O2]: https://openrouter.ai/openai/gpt-oss-120b:free "OpenRouter GPT-OSS 120B free model and tools; read 2026-10-24"
[O3]: https://openrouter.ai/docs/guides/routing/routers/free-router "OpenRouter free router and tool filtering; read 2026-10-24"
[O4]: https://openrouter.ai/docs/api_reference/limits "OpenRouter limits; numerical RPM table was not rendered by web extraction; read 2026-10-24"
[O5]: https://openrouter.ai/docs/guides/privacy/data-collection "OpenRouter and upstream prompt-data controls; read 2026-10-24"
[O6]: https://openrouter.ai/docs/faq "OpenRouter FAQ on published provider TTFT; read 2026-10-24"
[O7]: https://openrouter.ai/docs/api_reference/authentication "OpenRouter key authentication and creation; read 2026-10-24"
[B1]: https://inference-docs.cerebras.ai/support/rate-limits "Cerebras trial quotas, payment verification, expiration, no permanent free tier; read 2026-10-24"
[B2]: https://www.cerebras.ai/pricing "Cerebras pricing: $5 starter credit and pay-as-you-go; read 2026-10-24"
[H1]: https://docs.github.com/en/github-models "GitHub Models retirement notice; read 2026-10-24"

Coverage limit: public documents only; account-specific quotas, key issuance from India, free endpoint availability, semantic tool reliability, and actual TTFT remain unverified. No claims rely on remembered historical limits.
