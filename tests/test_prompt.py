from app.providers.base import build_description_prompt


def test_prompt_without_scene_context_or_names_says_not_to_guess():
    prompt = build_description_prompt()
    assert "Do not speculate about who any person is by name" in prompt
    assert "normal, unchanging view" not in prompt


def test_prompt_with_scene_context_steers_toward_whats_different():
    prompt = build_description_prompt(scene_context="a grassy yard with a gravel path")
    assert "a grassy yard with a gravel path" in prompt
    assert "rather than re-describing the fixed background" in prompt


def test_prompt_with_known_people_names_them_instead_of_guessing():
    prompt = build_description_prompt(known_people=["Cathleen Murphy"])
    assert "Cathleen Murphy" in prompt
    assert "Do not speculate about who any person is by name" not in prompt


def test_prompt_combines_scene_context_and_known_people():
    prompt = build_description_prompt(
        known_people=["Cathleen Murphy"], scene_context="a grassy yard with a gravel path"
    )
    assert "Cathleen Murphy" in prompt
    assert "a grassy yard with a gravel path" in prompt
