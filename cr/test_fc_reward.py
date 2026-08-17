import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc_reward import evaluate_completion, score_call, fc_reward

# exact match
ans = [{"id": 0, "name": "ACTION_SET_ALARM", "arguments": {"EXTRA_HOUR": 7, "EXTRA_MINUTES": 30}}]
r = evaluate_completion('[{"name":"ACTION_SET_ALARM","arguments":{"EXTRA_HOUR":7,"EXTRA_MINUTES":30}}]', ans)
assert abs(r - 1.0) < 1e-6, r
print("PASS exact ->", r)

# wrong args (name right, args wrong) -> 0.4
r = evaluate_completion('[{"name":"ACTION_SET_ALARM","arguments":{"EXTRA_HOUR":9,"EXTRA_MINUTES":0}}]', ans)
assert abs(r - 0.4) < 1e-6, r
print("PASS wrong args ->", r)

# wrong name -> 0.0 (name wrong, and args don't align to expected)
r = evaluate_completion('[{"name":"ACTION_SET_TIMER","arguments":{"EXTRA_HOUR":7,"EXTRA_MINUTES":30}}]', ans)
assert r == 0.0, r
print("PASS wrong name ->", r)

# invalid json -> 0
r = evaluate_completion("I will set an alarm.", ans)
assert r == 0.0, r
print("PASS invalid ->", r)

# partial chain: 2 expected, 1 correct
ans2 = [{"id":0,"name":"A","arguments":{"x":1}},{"id":1,"name":"B","arguments":{"y":2}}]
r = evaluate_completion('[{"name":"A","arguments":{"x":1}},{"name":"WRONG","arguments":{"z":9}}]', ans2)
assert abs(r - 0.5) < 1e-6, r
print("PASS partial chain ->", r)

# missing optional arg (None in expected, absent in gen)
ans3 = [{"id":0,"name":"send_email","arguments":{"to":["a@b.com"],"subject":"hi","body":"x","cc":None,"attachments":None}}]
r = evaluate_completion('[{"name":"send_email","arguments":{"to":["a@b.com"],"subject":"hi","body":"x"}}]', ans3)
assert abs(r - 1.0) < 1e-6, r
print("PASS optional none ->", r)

# think/answer wrapper with embedded json
r = evaluate_completion('<think>I will set an alarm.</think>\n<answer>\n[{"name":"ACTION_SET_ALARM","arguments":{"EXTRA_HOUR":7,"EXTRA_MINUTES":30}}]\n</answer>', ans)
assert abs(r - 1.0) < 1e-6, r
print("PASS wrapped ->", r)

# batch
comps = [
    '[{"name":"ACTION_SET_ALARM","arguments":{"EXTRA_HOUR":7,"EXTRA_MINUTES":30}}]',
    "garbage",
]
rewards = fc_reward(None, comps, None, answers=[ans, ans])
assert abs(rewards[0]-1.0) < 1e-6 and rewards[1] == 0.0, rewards
print("PASS batch", rewards)
print("ALL_FC_TESTS_PASSED")
