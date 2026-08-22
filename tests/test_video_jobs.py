import threading

from backend.video_jobs import VideoJobRunner


def test_video_job_reports_frame_progress_and_result():
    runner = VideoJobRunner(workers=1)
    started = threading.Event()
    release = threading.Event()

    def work(report):
        report("detect_landmarks", 30, 3, 10, "顔と目元を検出しています")
        started.set()
        assert release.wait(timeout=2)
        report("compose", 80, 8, 10, "各フレームを合成しています")
        return {"session_id": "video-session", "frame_count": 10, "composed": True}

    job_id = runner.submit("video-session", work)
    assert started.wait(timeout=2)
    running = runner.get(job_id)
    assert running is not None
    assert running.to_dict()["status"] == "running"
    assert running.to_dict()["processed_frames"] == 3
    assert running.to_dict()["total_frames"] == 10
    assert running.to_dict()["phase"] == "detect_landmarks"

    release.set()
    runner.wait(job_id, timeout=2)
    done = runner.get(job_id)
    assert done is not None
    assert done.to_dict()["status"] == "done"
    assert done.to_dict()["progress"] == 100
    assert done.to_dict()["result"] == {"session_id": "video-session", "frame_count": 10, "composed": True}


def test_video_job_runner_prunes_completed_jobs_and_futures():
    runner = VideoJobRunner(workers=1, max_retained_jobs=2)
    job_ids = []
    for index in range(3):
        job_id = runner.submit(f"session-{index}", lambda _report, i=index: {"index": i})
        runner.wait(job_id, timeout=2)
        job_ids.append(job_id)

    assert runner.get(job_ids[0]) is None
    assert runner.get(job_ids[1]) is not None
    assert runner.get(job_ids[2]) is not None
    assert runner._futures == {}
