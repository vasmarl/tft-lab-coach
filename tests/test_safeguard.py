import safeguard


def test_blocks_when_league_game_running():
    procs = ["explorer.exe", "chrome.exe", "League of Legends.exe", "LeagueClientUx.exe"]
    assert safeguard.game_running(procs) is True


def test_league_client_alone_is_fine():
    procs = ["explorer.exe", "LeagueClientUx.exe", "LeagueClient.exe", "RiotClientServices.exe"]
    assert safeguard.game_running(procs) is False


def test_window_filter_hides_riot_and_league():
    windows = [
        (1, "TFT guide - YouTube - Google Chrome", "chrome.exe", 10),
        (2, "League of Legends (TM) Client", "League of Legends.exe", 11),
        (3, "League of Legends", "LeagueClientUx.exe", 12),
        (4, "Riot Client Main", "RiotClientUx.exe", 13),
        (5, "Some window", "LeagueClientUxRender.exe", 14),
        (6, "Riot Client", "Riot Client.exe", 15),
        (7, "Teamfight Tactics - VLC media player", "vlc.exe", 16),
        (8, "My own coach panel", "TFTCoach.exe", 99),
        (9, "   ", "notepad.exe", 17),
        (10, "Lobby", "RiotClientServices.exe", 18),
        (11, "Lobby", "LeagueClient.exe", 19),
        (12, "Program Manager", "explorer.exe", 20),
    ]
    kept = safeguard.filter_windows(windows, own_pid=99)
    labels = [safeguard.label(w) for w in kept]
    assert labels == ["TFT guide - YouTube - Google Chrome [chrome.exe]"]
