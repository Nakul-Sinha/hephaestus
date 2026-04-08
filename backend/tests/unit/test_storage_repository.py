from backend.storage import InMemoryIncidentRepository, get_incident_repository


def test_repository_defaults_to_in_memory() -> None:
    get_incident_repository.cache_clear()
    repository = get_incident_repository()
    assert isinstance(repository, InMemoryIncidentRepository)
