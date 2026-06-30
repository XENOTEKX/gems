# STRIP MANIFEST — tree/gpu/gpu_lnl_intree.cu (3106 lines @ wip-extract 79891f57)

Produced by strip-classifier agent (a5515816407c3e501), 2026-06-30. Execute then red-team + p1f build.

## KEEP (LIVE — never delete)
- `kj_derv_fused` (437–455): launched 2686,2698,2753,2761,2840,2851,2875
- `kj_pre` (489–504): LIVE via gpu_allbranch_upper_check@1833 + JOLT reopt@2808 (only the `eigUpper` SELECTOR is removed)
- `k1_node`, `k1_node_prod`, `kj_pre_node`, `make_pmat`, `g_rscale`, `gb_valall/d_valall`, all 2D pars kernels
- ts_reopt_mcs/vp counter DECLS (2457–2458) + increments @2652(setVal),2852(legacy) — keep (harmless) OR remove increments too
- g_jdiag/JOLT_DIAG (2451 etc.) — NOT a strip target, keep (adjacent to g_lbfgs_init@2452)

## DELETE (DEAD, env-gated OFF) — content-based edits; promote the noted `else`
T1 async: g_ts_async/_check(668-669), streams+pin+valpool substrate(677-718), comment(658-667), kj_derv_fused_args(456-483).
  Promote-else dispatch: Site A(2034-41 →`const int S=1`), Site C(2254-2356 3-way→serial else), Site E(2742-63→legacy),
  Site F(2818-54→legacy), Site G(2864-76→legacy), Site H(2884-88 del), perPtnDoubles term(2047), valBlk/tsS(2543-52).
T2 batchfold: kernels screen_node1/2_batch+screen_k2_batch(240-341), globals(670-676), gb_*batch(653),
  allocs(2076-2097), batch arm(2295-2331, folded into Site C).
T3 L-BFGS: g_lbfgs_*(2418-2424), init line(2452 only), state decls(2955-2961), pair block(2965-2980),
  direction arm(3026-3056 → promote diagonal-LM else 3057-3079).
T4 1D pars: kernels k_pars_combine_to_arena + k_pars_score_indexed (NON-_2d)(811-907), use1d(1047-48),
  if(use1d) arm(1049-1078 → promote 2D else).
T5A kcount: g_kcount+counters(1999-2005), guards 2206,2216,2357,2359-2367,2809,3102-3104.
T5B GPUREDUCE: gb_sptnfreq/sredpart(650), TS_GPURED+bufs(2136-2146), reduceInto branch(2148-2152→host-Kahan default),
  upload(2197), GBmax(2143).
T5C eigUpper: eigUpper(2159-2161); collapse if(eigUpper)…else→else at 2166-2178, 2339-2343 (launchMove 2231-2235 dies w/ Site C).
T6 dead kernels: kj_theta(407-413), kj_derv(414-430), kj_ratenum(575-582). d_theta variable does not exist (comments only @432,2556,2811).

## DANGER (verbatim from agent)
1. kj_pre LIVE (1833,2808) — keep kernel, remove only eigUpper selector.
2. kj_derv_fused LIVE — don't confuse with kj_derv(dead)/kj_derv_fused_args(async).
4. launchMove(2221-2244)+gatherRow(2247-2249) lambdas die after async+batchfold arms gone — delete both.
5. ts_reopt_mcs+= @2652(setVal LIVE),2852(legacy LIVE): keep decls 2457-2458 OR remove increments too.
6. reoptEdges/reoptChkOk(2800,2847,2855) read only by async _check(2888) — dead after Site H; delete or leave harmless.
7. g_jdiag@2451 NOT a target — delete only g_lbfgs_init@2452.
8. const int S=1 keeps 2068-2073 byte-identical — don't strip S* factors.
9-11. node-space upper machinery, gb_valall, k1_node all LIVE — kept by promoting eigUpper/batchfold else-branches.

Verification after strip: grep proves zero live refs to each removed symbol; all KEEP symbols still present; p1f GPU build+smoke PASS; p1e unaffected (GPU-only file).
