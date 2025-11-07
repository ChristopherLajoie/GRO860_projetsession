from flappy.physics import collides


def test_collides_detects_overlap():
    bird = (10.0, 10.0, 20.0, 20.0)
    pipes = ((15.0, 0.0, 30.0, 40.0), (100.0, 200.0, 20.0, 100.0))
    assert collides(bird, pipes)


def test_collides_detects_separation():
    bird = (0.0, 0.0, 10.0, 10.0)
    pipes = ((20.0, 0.0, 5.0, 30.0), (20.0, 50.0, 5.0, 30.0))
    assert not collides(bird, pipes)
