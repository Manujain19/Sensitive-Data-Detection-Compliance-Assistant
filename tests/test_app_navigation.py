from streamlit.testing.v1 import AppTest

import app


def test_landing_get_started_buttons_open_signup_form():
    app = AppTest.from_file("app.py", default_timeout=10)
    app.run()

    app.button[1].click().run()

    assert [item.label for item in app.text_input] == [
        "Full name",
        "Email",
        "Password (min 8 chars)",
    ]


def test_landing_login_buttons_open_login_form():
    app = AppTest.from_file("app.py", default_timeout=10)
    app.run()

    app.button[0].click().run()

    assert [item.label for item in app.text_input] == ["Email", "Password"]


def test_google_oauth_state_is_stateless_and_tamper_resistant():
    state = app.create_google_oauth_state()

    assert app.validate_google_oauth_state(state)
    assert not app.validate_google_oauth_state(state + "x")
    assert not app.validate_google_oauth_state("")
