"""Final report narrative; all result-dependent values are read from calculation files."""
from report_values import (R, WEIGHTED_ROWS, CALIBRATION_ROWS, POLICY_ROWS, CLASS_ROWS, COHORT_ROWS, MATCHED_ROWS, SPLIT_ROWS)
def p(text): return {'type':'p','text':text}
def h(text): return {'type':'h','text':text}
def fig(name,height,caption): return {'type':'image','name':name,'height':height,'caption':caption,
    'directory':'archive/historical_research/figures' if name == 'unlabeled_comparison' else 'figures'}
def table(headers,rows,widths=None):return {'type':'table','headers':headers,'rows':rows,'widths':widths}

REPORT_PAGES=[
{'title':'ReAction','subtitle':'Behavioral response prediction | Final submission | 1 October 2026','blocks':[
 {'type':'callout','text':'DECISION: Do not deploy for live intervention decisions. Complete source validation, independent evaluation and governance review first.'},
 p('The assignment asks for **five outcome probabilities for each employee-situation pair**. The recommended intervention is an input. The forecast must help identify unsuitable default recommendations; it does not establish that changing the intervention will improve outcomes.'),
 table(['Assignment requirement','Where this submission answers it'],[
 ['1. Audit and representation','Data quality, missingness, feature contract and collection questions.'],
 ['2. Response prediction','Section 2: model, training strategy and supervised/unlabeled comparison; code in src/.'],
 ['3. Evaluation','Calibration, discrimination, age/domain slices and selection bias.'],
 ['4. Deployment memo','Serving, monitoring, delayed feedback, feedback loops and EU governance.']], [.30,.70]),
 p('**Inputs:** engagement questionnaire, OCEAN personality scores, optional wearable aggregates, and event context. The five intervention choices are peer workshop, self-paced video, one-to-one coaching, simulation exercise and reading pack.'),
 fig('pipeline',150,'Selected pipeline. All fitted transformations are learned inside the training boundary.'),
 p('**Output categories:** **completed effective**, **completed ineffective**, **partial**, **dropped out** and **declined**. Probabilities sum to 100% for each situation.'),
 {'type':'kpis','items':[('20,000','employees'),('119,498','situations'),('4.68%','with recorded outcomes')]},
 p(f'**Selected model:** temperature-scaled LightGBM without age or wellness inputs. Log loss is **{R["log_loss"]:.3f}**, versus **{R["prior_loss"]:.3f}** for class frequencies. These are development estimates; independent validation remains outstanding.')
]},
{'title':'1. Data audit and representation','subtitle':'Prediction unit: one employee in one situation','blocks':[
 p('The data contain 20,000 employees, 119,498 situations and 5,587 observed outcomes. Validated person and situation keys preserve all 113,911 situations without feedback. Joins are many-to-one for employee profiles and one-to-one for recorded responses. Identifiers are used for alignment and employee grouping only. An absent response is unknown, not failure or a sixth category. Repeated events from one employee are dependent observations.'),
 p('Observed outcomes are imbalanced: **dropout** has 1,522 records, **effective completion** 1,496, **ineffective completion** 905, **decline** 887 and **partial completion** 777. These counts describe feedback collection, not population outcome prevalence. The outcome rubric also requires validation: “**effective**” and “**partial**” must have consistent definitions across modules and raters before they support employee-facing decisions.'),
 p(f'Missingness is substantial. Burnout is absent for 30.37% of employees, OCEAN scores for 6.3-6.6%, and questionnaire items for 7.1-7.9%. Wearables are structurally absent for 49.51% who did not opt in; further gaps occur among participants. HRV missingness rises from {R["hrv_young"]:.1%} below age 30 to {R["hrv_old"]:.1%} at 50+; {R["hrv_optin"]:.1%} of opt-ins still lack HRV. These patterns separate structural opt-out, demographic selection and measurement gaps. They challenge simple missingness assumptions without identifying the unobserved-data mechanism. Missing physiology is never zero.'),
 p('Four questionnaire pairs match whenever both values are observed: q01/q19, q02/q20, q09/q17 and q10/q18. Their missingness differs. Keep both raw items and their missing indicators until item provenance is resolved: dropping one would discard independently available measurements. Do not interpret duplicated items as independent psychometric evidence. HRV correlates +0.912 with sleep variability; sleep variability correlates -0.772 with burnout. Five employees have negative active minutes. Research models replace those invalid values with missingness and retain an error flag. Item keys, units and composite derivations must be checked; correlations alone do not justify recoding.'),
 p(f'Feedback is selective. Formal training represents about half of all situations but three quarters of labels. Labeled and unlabeled records also differ on physiology, conscientiousness and burnout, with standardized mean differences around 0.19-0.23. A model of feedback observation reaches AUC {R["observation_auc"]:.3f} on employee-held-out data. This demonstrates observable selection; it cannot establish the distribution of unobserved outcomes.'),
 p('Timing blocks deployment. Outcome timestamps extend to May 2050; 4,833 fall after the audit day. Event and feature-as-of timestamps are absent. Confirm whether dates are synthetic, shifted or incorrect. The labels remain usable for this stated modeling exercise, but no temporal deployment backtest is justified. Removing future dates without understanding their origin could introduce further bias.'),
 p('The selected feature contract combines named event context, tenure, job/region, 24 questionnaire items and OCEAN. It excludes age, burnout, opt-in, wearables and anonymous context. Training-fold medians, missing indicators, scaling and categorical encoding are serialized for inference. Excluded fields are not required at serving time. Age proxies may remain; feature exclusion does not establish fairness.'),
]},
{'title':'3. Evaluation scope and representativeness','subtitle':'Development estimates under selective feedback','blocks':[
 p(f'The study uses **{R["labeled_events"]:,} labeled events from {R["labeled_people"]:,} employees**, using master seed **{R["seed"]}**. Five outer folds hold out entire employees; three inner folds fit scalar temperature calibration. Preprocessing is fitted inside each boundary. Model selection uses these development data, so the scores do not constitute an independent final test.'),
 p('The 3,318 labels are the historical 60% employee training allocation. The remaining 2,269 labels are excluded to preserve comparability with prior fits; their old tune/calibration/test names no longer imply those roles. No historical allocation is certified untouched. This sacrifices data efficiency and does not restore independence. Section 2 gives split counts and the supervised-versus-unlabeled comparison.'),
 fig('feedback_coverage',100,'Feedback coverage differs by more than threefold across domains. Collection questions and responsible owners appear in Section 4.'),
 h('Representativeness sensitivity'),
 p('An employee-held-out observation model estimates the probability that feedback is recorded. Inverse-probability weighting assumes conditional independence of outcome and observation, sufficient coverage and an adequate observation model. Those assumptions are unverified.'),
 table(['Evaluation weighting','Log loss','Brier','Effective events'], WEIGHTED_ROWS, [.49,.16,.15,.20]),
 p('Weighting is a sensitivity check, not corrected ground truth. Effective counts measure weight concentration, not independent employees. The next collection must sample feedback independently of model flags and track invitations, reminders, maturity and nonresponse.')
]},
{'title':'3. Calibration and discrimination','subtitle':'A fitted calibrator is not a calibration guarantee','blocks':[
 table(['Probability metric','Before scaling','After scaling'], CALIBRATION_ROWS, [.52,.24,.24]),
 p(f'Temperature scaling slightly worsens the point estimates. Its paired log-loss change is {R["calibration_delta"]}; Brier change is {R["calibration_brier"]}. Both intervals span zero. Keep the declared procedure without selecting a replacement on these outer folds. Temperature-scaled does not mean calibration is established. The prior baseline has little binned calibration error but no individual discrimination.'),
 fig('calibration',244,'Solid markers: at least 100 events and 50 employees. Error bars: conditional 95% employee-bootstrap intervals. Hollow markers: sparse bins; descriptive only. The diagonal denotes agreement.'),
 {'type':'kpis','items':[(f'{R["accuracy"]:.1%}','accuracy'),(f'{R["macro_f1"]:.3f}','macro-F1'),(f'{R["auc"]:.3f}','macro OVR AUC')]},
 p(f'Balanced accuracy is **{R["balanced_accuracy"]:.3f}**. Several classes have weak argmax recall. Their probability quality and operational use need class-specific review; low recall alone does not prove that the probabilities are unusable.'),
 p('Intervals condition on the fitted models and bin definitions. They do not cover all model-selection or refitting uncertainty. Sparse all-success or all-failure bins cannot support reliability claims.')
]},
{'title':'3. Class and cohort diagnostics','subtitle':'Probability quality and discrimination require separate interpretation','blocks':[
 table(['Outcome','n','Precision','Recall','F1'],CLASS_ROWS,[.40,.12,.16,.16,.16]),
 p('Argmax recall is below 9% for ineffective completion, partial completion and decline. The model mainly predicts effective completion or dropout as its most likely outcome. This restricts label-based use; useful probability forecasts still require classwise reliability and proper-score evidence.'),
 table(['Age / domain','n','Log loss [95% interval]','AUC','Top ECE'],COHORT_ROWS,[.24,.10,.34,.14,.18]),
 p('AUC is macro one-versus-rest discrimination; ECE is ten-bin top-label calibration error. Log-loss intervals resample employees, conditional on the existing fits. AUC/ECE are descriptive point estimates without uncertainty intervals. The missing-age and external-event groups are particularly imprecise. Differences in outcome mix and selective feedback prevent interpreting raw loss gaps as a fairness verdict. Class/cohort calibration bins and employee counts are retained in results/model/.'),
]},
{'title':'3. Cohorts and decision policy','subtitle':'Unequal performance and excessive review volume block release','blocks':[
 fig('cohort_performance',120,'Selected model. Bars show conditional 95% employee-bootstrap log-loss intervals; n denotes labeled situations. These estimates do not include refit or model-selection uncertainty.'),
 p(f'The **{R["worst_age"]}** cohort has the highest age-group loss despite age being excluded. External events have {R["external_n"]:,} labeled examples. Age-by-domain, intervention and opt-in diagnostics are included in the result files. These analyses do not establish fairness.'),
 p(f'Opt-out and opt-in log losses are **{R["optout_loss"]:.3f}** and **{R["optin_loss"]:.3f}**. Restricted audit attributes must remain separate from ordinary serving logs. Removing age or wellness predictors does not remove every proxy or collection effect.'),
 h('Tested shadow policy'),
 p('Illustrative assumptions: flag the default intervention when **P(completed effective) < 30%**, with review capacity set to 20%. Neither value is product-approved. The policy groups all other outcomes as “**not effective completion**” for diagnosis; it does not assume equal business costs and does not rank alternative interventions.'),
 fig('review_capacity',85,'Labeled development cases only. The illustrative threshold exceeds assumed 20% capacity and is rejected for live use.'),
 table(['Policy diagnostic','Result'], POLICY_ROWS, [.76,.24]),
 p(f'**{R["near_events"]:,} labeled events from {R["near_people"]:,} employees** lie within 10 percentage points of the threshold. Operational workload must count all forecasts. Product owners must approve costs, capacity and thresholds before prospective evaluation; retuning on evaluation outcomes cannot establish success.')
]},
{'title':'4. Deployment design memo','subtitle':'A controlled shadow workflow with explicit ownership','blocks':[
 p('Release requires validated timestamps, outcome definitions, independent evidence and governance approval. Start with a silent shadow run on scheduled L&D events. A daily EU-hosted batch job must join approved person/context snapshots, validate the schema, load the versioned artifact and store five probabilities with situation ID, model hash, forecast time and policy version. Keep the live recommendation unchanged. Do not expose individual shadow scores to managers.'),
 p(f'The selected service needs no age, burnout, opt-in or wearable inputs. Keep audit attributes in a separate restricted store. Warm local p95 inference, including cleaning and transformation, measured {R["p95_one"]:.1f} ms for one event, {R["p95_hundred"]:.1f} ms for 100 and {R["p95_thousand"]:.1f} ms for 1,000 over 30 repetitions. These timings exclude retrieval, transport, cold start and concurrency. Proposed service targets are p95 below 100 ms and 99.5% availability; the platform owner must verify them under load.'),
 p('Schema or join failures must preserve the default recommendation and log a reason code. Enforce measurement bounds and event-category values; impute legitimate missing measurements. New job families or regions remain encodable and trigger monitoring. Alert the platform owner if failures exceed 1% or p95 exceeds 100 ms over 15 minutes. The data owner must investigate new categories or missingness increases above five percentage points. Monitor weekly probability distributions, flag volumes and feature drift against the frozen reference; changes above five percentage points prompt review. Track maturity, coverage and nonresponse by cohort. These signals require investigation, not automatic retraining. Maintain versioned artifacts and a tested rollback procedure.'),
 p('Freeze forecasts before feedback is available. Close each cohort no earlier than eight weeks after its last event. Record invitation, reminder, response and maturity timestamps; pending feedback must remain unknown. Evaluate loss, Brier, reliability and age/domain slices only after planned support and coverage are met. The model owner must approve retraining and a newly frozen comparison. Late feedback belongs in a separately versioned review, never an overwritten forecast.'),
 p('Collect feedback across the population independently of model flags. Record the actual intervention, recommendation policy, human overrides and reasons. Otherwise, the model changes the data that train its successor and can reinforce its own errors. Alternative matching requires a separate study. Employees need a clear purpose explanation, a correction route, and a way to challenge consequential use with meaningful human oversight.'),
 p('Before employee shadow processing, the controller must document whether a GDPR Article 35 DPIA is required and complete it where required, with DPO advice and employee-representative consultation [4]. Approve purpose, lawful basis, minimization, access and retention. A six-month restricted retention period is proposed, subject to review and deletion-rights handling. Employment power imbalance limits reliance on consent [1]. Any health-related inputs require separate assessment of applicable special-category conditions [2]. Assess intended use against AI Act Annex III employment provisions. If it falls within Annex III and profiles people, Article 6(3) treats it as high-risk; the L&D label or human review does not itself create an exemption [3]. Legal counsel must verify applicable obligations and dates. Statistical acceptance does not supply legal approval.'),
 fig('outcome_timeline',96,'Labels arrive 4-8 weeks after the event. Pending feedback remains unknown throughout the cycle.')
]},
{'title':'4. Release requirements','subtitle':'Required evidence, accountable owners and reproducible delivery','blocks':[
 table(['Unresolved question','Required action / owner'],[
 ['Are dates valid and features available before each event?','Certify event and feature timestamps; explain future dates. Data engineering.'],
 ['What constitutes **effective completion** or **partial completion**?','Approve a common rubric and double-code a sample. L&D.'],
 ['Are psychometric fields correctly defined?','Check item keys, duplicate pairs, units and derived scores. Measurement owner.'],
 ['Who is invited to provide feedback, and when?','Document sampling, reminders and maturity. L&D operations.'],
 ['What decision is permitted and affordable?','Approve costs, review capacity, legal purpose and access. Product + DPO.']], [.46,.54]),
 h('Independent evaluation'),
 p('No independent final cohort has been evaluated. For future validation, recruit employees absent from the supplied population and record features before forecasts and forecasts before events. Freeze model, code, input and prediction hashes; reject changed records and evaluate only after outcomes mature.'),
 p('Proposed minimums are **2,000 labeled events, 1,000 employees and 100 examples per outcome**. Each required age/domain slice needs at least 100 events, 50 employees and 80% feedback coverage. Outcome metrics remain withheld until those minimums are met.'),
 p('Require at least **0.002 lower log loss** than the frozen class-prior comparator, with a paired interval below zero; Brier deterioration must remain within 0.005. Supported class/cohort reliability intervals must remain within +/-10 percentage points. The policy must fit review capacity. Approve these proposed tolerances and complete a precision/power review before recruitment.'),
 h('Submission contents'),
 p('Submit the complete folder. README.md documents reproduction; data/ holds unchanged inputs; src/ contains Tasks 1-2; results/ contains models and evidence; scripts/ rebuilds the report. This report covers all four tasks. Include archive/historical_research/ for the historical comparisons cited here.'),
 p('**Reproducibility:** use seed 20261001. Input/split hashes are in results/splits/manifest.json; the artifact hash is in results/model/model_card.json. CSV/JSON results retain full precision; report values are rounded.'),
 h('Sources'),
 p('[1] [EDPB: lawful processing and employee consent](https://www.edpb.europa.eu/sme/be-compliant/process-personal-data-lawfully_en).'),
 p('[2] [European Commission: legal grounds and sensitive data](https://commission.europa.eu/law/law-topic/data-protection/information-business-and-organisations/legal-grounds-processing-data_en).'),
 p('[3] European Commission AI Act Service Desk: [Article 6](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-6) and [Annex III](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3).'),
 p('[4] [EDPB: data protection impact assessment](https://www.edpb.europa.eu/topics/accountability-and-compliance-tools/data-protection-impact-assessment_en).')
]}
]

# Task 2 is included in the same canonical report source.
TASK2_PAGES=[
{'title':'2. Model and training strategy','subtitle':'Select probability quality; report classification separately','blocks':[
 p(f'The study uses **{R["labeled_events"]:,} labeled events from {R["labeled_people"]:,} employees**, using master seed **{R["seed"]}**. Five outer folds hold out entire employees; three inner folds fit scalar temperature calibration. Preprocessing is fitted inside each boundary. Model selection uses these development data, so the scores do not constitute an independent final test.'),
 table(['Historical split','Labels','Current role'],SPLIT_ROWS,[.20,.14,.66]),
 p('Retain the historical training allocation for reproducible comparisons with earlier experiments and avoid silently changing both data and method during this review. The other 2,269 labels remain excluded, even though prior development exposure prevents treating them as independent. This is a conservative artifact-maintenance choice, not an optimal use of scarce labels. After the protocol is finalized, an all-label refit would require a new version and external validation. Existing predictions must not be presented as its evaluation.'),
 p('Tree models use 100 boosting rounds, seven leaves, at least 40 samples per leaf, learning rate 0.05 and L2 penalty 10. The logistic baseline uses C=0.1. The class-frequency baseline estimates five probabilities from each training fold. All comparisons preserve employee separation.'),
 fig('model_comparison',237,'Calibrated development scores. Lower is better. Teal marks the selected model; orange marks models using wellness or undefined context features.'),
 p(f'Questionnaires and personality provide the main measured gains. Adding questionnaires reduces log loss by **{R["questionnaire_gain"]:.3f}**; adding personality reduces it by **{R["personality_gain"]:.3f}**. Context alone does not outperform class frequencies. Restoring the 50 anonymous features worsens log loss by **{R["anonymous_delta"]:.3f}**.'),
 p(f'Wellness improves log loss by **{R["wellness_gain"]:.3f}** relative to the age-inclusive personality model, but its Brier interval includes zero. It is excluded pending purpose and legal review. The anonymous fields are excluded pending definitions and feature timing.'),
 p('Among eligible models, choose the least demanding feature set whose paired upper interval stays within **0.010 log loss** and **0.005 Brier** of the best eligible model. The age-omitting personality model meets these development margins. They are proposed engineering tolerances, not a confirmatory non-inferiority result.'),
 p(f'The final artifact is fitted on the training labels with temperature **{R["temperature"]:.3f}**. It returns the same five outcomes in a fixed order. Macro-F1 is diagnostic; it does not override the probability objective.')
]},
{'title':'2. Historical unlabeled-data experiments','subtitle':'Use unlabeled inputs without inventing outcomes','blocks':[
 p('Unlabeled situations support coverage analysis, missingness analysis and three training experiments: denoising features, SCARF-style contrastive features and soft pseudo-labels. Held-out employees are excluded from each fitting pool. These historical research arms include age and wellness and were not separately calibrated. Their results apply to that configuration. The selected-contract comparison in the preceding section addresses this limitation; its evidence supports retaining supervised learning.'),
 p('These abandoned research branches, their supervised controls, code and saved results are preserved in archive/historical_research/. They are retained as historical evidence, separate from the active selected-contract follow-up.'),
 p('Each method has a matched labeled-only control. This separates the effect of additional unlabeled inputs from the effect of changing the learner or representation. Soft targets retain all five teacher probabilities; their total training weight is capped at 30% of the real-label weight.'),
 fig('unlabeled_comparison',185,'Mean log-loss changes with conditional 95% employee-bootstrap intervals across ten development repetitions. Intervals crossing zero do not establish improvement.'),
 p(f'Soft pseudo-labels change log loss by **{R["soft_matched"]:+.4f}** against their matched control and **{R["soft_calibrated"]:+.4f}** against the calibrated supervised control. The latter 95% interval is **[{R["soft_lower"]:+.4f}, {R["soft_upper"]:+.4f}]**, {R["soft_interval_status"]}. Negative changes favor pseudo-labels. Research arms were not separately calibrated; this comparison cannot establish superiority after common calibration.')
]}
]

TASK2_PAGES.append({
 'title':'2. Supervised-only versus unlabeled-data training','subtitle':'Selected inputs, identical folds and the same calibration procedure','blocks':[
 p('The follow-up uses the selected age/wellness-free contract and repeat 0 of the same five employee outer folds. Each arm fits its own scalar temperature on the same three inner employee folds. SSL fitting excludes both outer and inner held-out employees from labeled and unlabeled pools. The supervised arm exactly reproduces the original selected-model predictions. Method choice follows earlier research, so this remains post-review development evidence.'),
 {'type':'callout','text':'TRAINING DECISION: retain supervised learning. Adding unlabeled pseudo-labels does not establish an improvement over the supervised baseline.'},
 table(['Primary comparison (calibrated)','Log loss','Brier'],[MATCHED_ROWS[0], ['Supervised + unlabeled pseudo-labels', *MATCHED_ROWS[2][1:]]],[.62,.19,.19]),
 p(f'Adding unlabeled pseudo-labels changes log loss by **{R["selected_ssl"]}** relative to supervised-only training. Lower is better; the interval crosses zero, so this does not establish an improvement. It also misses the proposed 0.002 minimum mean gain. Unlabeled inputs still support coverage and missingness audits; they are not used to fit the final selected model.'),
 h('Supporting controls'),
 table(['Additional calibrated arm','Log loss','Brier'],[MATCHED_ROWS[1], MATCHED_ROWS[3]],[.62,.19,.19]),
 p(f'Against the matched labeled-only soft-target learner, the change is {R["selected_matched"]}. This isolates the effect of extra unlabeled inputs within that learner. It improves that control, but does not establish superiority over ordinary supervised learning.'),
 p(f'Adding event context to person-only supervised inputs changes log loss by {R["context_added"]}. This directly supports combining person and situation features, even though context alone is weak. Both arms use the same learner, folds and calibration procedure.'),
 p('Intervals are paired employee bootstraps conditional on saved fits. This one-repeat follow-up is separate from the historical ten-repeat research experiment and does not correct for historical method search. Training and reproduction code is src/compare_training_strategies.py; outputs are in results/revision/.')
 ]})

# Keep each logical section on a readable PDF page and put the primary comparison
# before the historical experiments. Both exports consume these same page objects.
training, historical, comparison = TASK2_PAGES
training_blocks = training['blocks']
training = {**training, 'blocks': training_blocks[:4]}
selection = {'title': '2. Feature selection and final model',
             'subtitle': 'Representation ablations and the selected calibrated artifact',
             'blocks': training_blocks[4:]}
TASK2_PAGES = [training, selection, comparison, historical]
REPORT_PAGES = REPORT_PAGES[:2] + TASK2_PAGES + REPORT_PAGES[2:]
