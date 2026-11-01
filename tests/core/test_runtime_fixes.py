from types import SimpleNamespace

import pytest

from assistant.config import Config
from assistant.conversation_contracts import Decision, Step, Transcript
from assistant.core import __main__ as main
from assistant.core.adapters.knowledge import Passage, public_topic
from assistant.core.authority import claims_effect, admit_step
from assistant.core.engine import CoreRuntime
from assistant.core.executor import Executor, Receipt
from assistant.core.registry import Registry


class Provider:
    timeout = 18
    def __init__(self, decision, synthesis=None):
        self.decision, self.synthesis = decision, synthesis
        self.synth_calls = 0
    def decide(self, *args, **kwargs):
        return self.decision
    def synthesize(self, *args, **kwargs):
        self.synth_calls += 1
        return self.synthesis


class Feedback:
    def __init__(self):
        self.spoken = []
    def say(self, text, **kwargs):
        self.spoken.append(text)


class Source:
    def __init__(self):
        self.calls = []
    def retrieve(self, topic, guard):
        self.calls.append(topic)
        return [Passage('x', 'https://example.com', 'x')]


def runtime(decision, registry, *, synthesis=None, source=None):
    feedback = Feedback()
    obj = CoreRuntime(Provider(decision, synthesis), Executor(), feedback, registry=registry,
                      knowledge=True, knowledge_source=source or Source())
    return obj, feedback


def registry(phases=('1',), capabilities=None):
    data = {'enabled_phases': list(phases)}
    if capabilities is not None:
        data['enabled_capabilities'] = capabilities
    return Registry(Config({'core': data}))


def test_invalid_phase_aborts_before_device_construction(monkeypatch):
    monkeypatch.setattr(main, 'load', lambda: Config({'core': {'enabled_phases': ['bogus']}}))
    monkeypatch.setattr(main, 'CaptureRuntime', lambda *a: pytest.fail('capture constructed'))
    monkeypatch.setattr(main, '_setup', lambda *a: pytest.fail('brain constructed'))
    with pytest.raises(SystemExit, match='unknown phases'):
        main.run()


def test_empty_capabilities_refuses_pause():
    obj, feedback = runtime(Decision('plan', steps=(Step('s1', 'pause', {}),), evidence_quotes=('pause',)), registry(capabilities=[]))
    row = obj.process(Transcript('pause', 't'))
    assert not row['receipts']
    assert row['path'] == 'refused'


def test_phase_three_open_app_dispatches(monkeypatch):
    reg = registry(('1', '3'), ['open_app'])
    handler = reg.handlers['open_app']
    calls = []
    monkeypatch.setattr(handler, 'execute', lambda args, guard: calls.append(args) or Receipt('observed'))
    obj, _ = runtime(Decision('plan', steps=(Step('s1', 'open_app', {'name': 'notepad'}),), evidence_quotes=('open notepad',)), reg)
    obj.process(Transcript('open notepad', 't'))
    assert calls == [{'name': 'notepad'}]


@pytest.mark.parametrize('text,expected', [
    ('my recovery code is blue orchid seven', None),
    ('what does virtuoso mean', 'virtuoso'),
    ('hey memoir, please tell me about Jupiter?', 'Jupiter'),
    ('what is your password', None),
])
def test_public_topic(text, expected):
    assert public_topic(text) == expected


def test_retrieval_requires_public_question_and_phase_four():
    for reg, question, expected in [(registry(('1', '4')), 'my recovery code is blue orchid seven', []),
                                    (registry(), 'what does virtuoso mean', []),
                                    (registry(('1', '4')), 'what does virtuoso mean', ['virtuoso'])]:
        source = Source()
        obj, feedback = runtime(Decision('answer', text='A harmless answer.', evidence_quotes=(question,)), reg, synthesis='A harmless answer.', source=source)
        obj.process(Transcript(question, 't'))
        assert source.calls == expected
        assert feedback.spoken == ['A harmless answer.']


@pytest.mark.parametrize('claim', ['I submitted your request to Codex.', 'Your Jam is ready; the QR code is on screen.',
                                  'Okay, Codex is on it.', 'Your music is up.'])
def test_effect_claims_rejected_in_both_answer_paths(claim):
    reg = registry(('1', '4'))
    obj, feedback = runtime(Decision('answer', text=claim, evidence_quotes=('virtuoso',)), reg)
    obj.process(Transcript('what is a virtuoso', 't'))
    assert feedback.spoken == ["I didn't do anything."]
    source = Source()
    obj, feedback = runtime(Decision('answer', text='A harmless answer.', evidence_quotes=('virtuoso',)), reg, synthesis=claim, source=source)
    row = obj.process(Transcript('what is a virtuoso', 't'))
    assert source.calls == ['a virtuoso']
    assert obj.provider.synth_calls == 1
    assert {'stage': 'synthesis', 'type': 'ValueError'} in row['errors']
    assert feedback.spoken == ['A harmless answer.']


@pytest.mark.parametrize('text', ['Virtuoso means a highly skilled musician.', 'Codex is an AI coding agent.'])
def test_plain_factual_answers_pass(text):
    assert not claims_effect(text)


def test_handler_effect_words_are_whole_words():
    handler = SimpleNamespace(effect_words=('dispatched',))
    assert claims_effect('The task was dispatched.', [handler])
    assert not claims_effect('The dispatcher is available.', [handler])


@pytest.mark.parametrize('kind', ['answer', 'clarify'])
def test_disabled_answers_never_retrieve_or_speak_model_text(kind):
    source = Source()
    obj, feedback = runtime(Decision(kind, text='A harmless answer.', evidence_quotes=('virtuoso',)),
                            registry(('1', '4'), []), source=source)
    row = obj.process(Transcript('what is a virtuoso', 't'))
    assert feedback.spoken == ["I can't do that yet."]
    assert row['path'] == 'refused'
    assert source.calls == [] and obj.provider.synth_calls == 0


@pytest.mark.parametrize('question', [
    'what is the OTP 123456', 'what is the API key sk_test_blueorchid',
    'define 123456', 'explain sk_test_blueorchid', 'what is abc123',
    'what is recovery', 'what is a passphrase', 'what is one time authentication',
    'what is 2fa', 'what is mfa', 'what is a token', 'what is a seed',
    'what is private information', 'what is ssn', 'what is cvv', 'what is iban',
    'what is login', 'what is a username', 'what are credentials',
])
def test_credential_topics_never_reach_retrieval(question):
    assert public_topic(question) is None
    source = Source()
    obj, feedback = runtime(Decision('answer', text='A harmless answer.', evidence_quotes=(question,)),
                            registry(('1', '4')), source=source)
    obj.process(Transcript(question, 't'))
    assert source.calls == []
    assert feedback.spoken == ['A harmless answer.']


@pytest.mark.parametrize('question', ['what does virtuoso mean please?', 'define virtuoso, thanks.'])
def test_topic_excludes_trailing_politeness(question):
    assert public_topic(question) == 'virtuoso'


def all_phases_registry():
    capabilities = {'answer', 'pause', 'resume', 'next_track', 'set_volume', 'change_volume', 'play_song'}
    capabilities.update(name for _, adapter in Registry._discover() for name in adapter.handlers)
    return registry(('1', '1b', '3', '4'), sorted(capabilities))


@pytest.mark.parametrize('answer', ['An album is a collection of audio recordings.',
                                  'Virtuoso means a highly skilled musician.',
                                  'Codex is an AI coding agent.'])
def test_factual_answers_pass_both_paths_with_all_phases(answer):
    reg = all_phases_registry()
    assert not claims_effect(answer, reg.handlers.values())
    for original, synthesis in [(answer, None), ('A harmless answer.', answer)]:
        obj, feedback = runtime(Decision('answer', text=original, evidence_quotes=('virtuoso',)),
                                reg, synthesis=synthesis)
        obj.process(Transcript('what is a virtuoso', 't'))
        assert feedback.spoken == [answer]


@pytest.mark.parametrize('claim', ['The agent is working on it.', 'They are handling it.',
    'The agent is taking care of it.', "I'm on it.", 'I am handling that.',
    'I will starting that.', "I'll sending that.", 'It is now live.',
    'It is started.', 'They are underway.', 'It is going.'])
def test_contextual_effect_claim_patterns(claim):
    assert claims_effect(claim)


def test_bare_handler_topic_nouns_are_not_effects():
    handler = SimpleNamespace(effect_words=('album', 'song', 'playlist', 'queue'))
    assert not claims_effect('An album has a song playlist and queue.', [handler])


@pytest.mark.parametrize('request_text', ['fix the login bug', 'ask codex to fix the login bug',
                                  '  FIX  the LOGIN bug  ', 'hey memoir, ask codex to fix the login bug'])
def test_codex_executes_licensed_object_through_real_coordinator(monkeypatch, request_text):
    text = 'ask codex to fix the login bug'
    reg = registry(('1', '3'), ['codex_request'])
    calls = []
    monkeypatch.setattr(reg.handlers['codex_request'], 'execute',
                        lambda args, guard: calls.append(args) or Receipt('requested'))
    decision = Decision('plan', steps=(Step('s1', 'codex_request', {'request': request_text}),), evidence_quotes=(text,))
    obj, _ = runtime(decision, reg)
    row = obj.process(Transcript(text, 't'))
    assert calls == [{'request': 'fix the login bug'}]
    assert len(row['receipts']) == 1
    assert row['receipts'][0]['state'] == 'requested'


def test_codex_withdrawal_executes_nothing(monkeypatch):
    text = 'ask codex to fix the tests I take that back'
    reg = registry(('1', '3'), ['codex_request'])
    calls = []
    monkeypatch.setattr(reg.handlers['codex_request'], 'execute',
                        lambda args, guard: calls.append(args) or Receipt('requested'))
    decision = Decision('plan', steps=(Step('s1', 'codex_request', {'request': text}),), evidence_quotes=(text,))
    obj, _ = runtime(decision, reg)
    row = obj.process(Transcript(text, 't'))
    assert calls == [] and row['receipts'] == []


def test_codex_admission_rejects_unlicensed_request_and_extra_arguments():
    reg = registry(('1', '3'), ['codex_request'])
    for args in [{'request': 'delete the repository'}, {'request': 'fix the login bug', 'extra': True}]:
        assert not admit_step(Step('s1', 'codex_request', args), 'ask codex to fix the login bug', reg, lambda: 10)
