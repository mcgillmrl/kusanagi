#!//usr/bin/env bash
ODIR=$1
EVAL_OPTS='-e cartpole.Cartpole -c cartpole.cartpole_loss -k n_trials 20 -k last_iteration 50'
for d in ${ODIR}/pilco_ssgp_rbfp_1*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p RBFPolicy
done
for d in ${ODIR}/mcpilco_dropoutd_rbfp_3*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p RBFPolicy
done
for d in ${ODIR}/mcpilco_dropoutd_mlpp_4*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p NNPolicy
done
for d in ${ODIR}/mcpilco_lndropoutd_rbfp_5*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p RBFPolicy
done
for d in ${ODIR}/mcpilco_lndropoutd_mlpp_6*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p NNPolicy
done
for d in ${ODIR}/mcpilco_dropoutd_dropoutp_7*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p NNPolicy
done
for d in ${ODIR}/mcpilco_lndropoutd_dropoutp_8*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p NNPolicy
done
for d in ${ODIR}/mcpilco_cdropoutd_dropoutp_9*/; do
    python kusanagi/shell/evaluate_policy.py $EVAL_OPTS -d $d -p NNPolicy
done