from app.memory.farm_memory import farm_memory_store

def test_farm_memory_retrieve():
    farm = farm_memory_store.get_farm("FARM_101")
    assert farm is not None
    assert farm["farmer_name"] == "Ramesh Patil"
    assert farm["location"]["district"] == "Pune"

def test_farm_memory_log_action():
    log = farm_memory_store.log_action("FARM_101", "Spray Action", "Applied Mancozeb 75 WP @ 2.5g/L")
    assert log["type"] == "Spray Action"
    farm = farm_memory_store.get_farm("FARM_101")
    assert len(farm["action_logs"]) > 0
