from openai import AzureOpenAI
from openai import APIStatusError, APIConnectionError, APITimeoutError, RateLimitError


def validate_endpoint_and_key(client: AzureOpenAI) -> tuple[bool, str | None]:
    """Return (ok, reason). Checks endpoint+api_version+key via models.list()."""
    try:
        _ = client.models.list()  # lightweight call; no deployment required
        return True, None
    except APIConnectionError as e:
        return False, f"Network error: {e}"
    except RateLimitError as e:
        return False, f"Rate limited: {e}"
    except APIStatusError as e:
        code = getattr(e, "status_code", None)
        return False, f"API error {code}: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def validate_deployment(client: AzureOpenAI, deployment_name: str) -> tuple[bool, str | None]:
    """Return (ok, reason). Verifies the given deployment exists and is callable."""
    if not deployment_name:
        return False, "Missing deployment name."
    try:
        client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        return True, None
    except APIStatusError as e:
        code = getattr(e, "status_code", None)
        if code == 404:
            return False, f"Deployment '{deployment_name}' not found for this resource."
        if code in (401, 403):
            return False, f"Not authorized to use deployment '{deployment_name}'."
        return False, f"API error {code}: {e}"
    except (APIConnectionError, APITimeoutError, RateLimitError) as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"



