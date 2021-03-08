from flask import *
import pytest

from run import createApp
from python_src.opts import get_parser

@pytest.fixture
def client():
    args = get_parser().parse_args()
    app = createApp(args)
    yield app.test_client()

@pytest.mark.parametrize("method, route, status_code", [
    ('GET','/',302),
    ('GET','/index',200),
    ('POST','/index',200),
    ('GET','/post',405),
    ('POST','/post',302),
    ('GET','/result',302)
])
def test_response_code_check(client, method, route, status_code):
    if method == 'GET':
        response = client.get(route)
    elif method == 'POST':
        response = client.post(route)
    else:
        raise Exception('HTTP request settings are incorrect.')
    assert response.status_code == status_code

@pytest.mark.parametrize("set_args, video_name, status_code", [
    ('--sample_video_dir ./sample_video', 'v_TennisSwing_g01_c01.avi', 302),
    ('--sample_video_dir ./sample_video_empty', 'v_TennisSwing_g01_c01.avi', 302),
    ('--sample_video_dir ./sample_video', 'nothing_video.avi', 302),
])
def test_post_to_result(set_args, video_name, status_code):
    args = get_parser().parse_args(set_args.split(' '))
    app = createApp(args)
    client = app.test_client()

    response = client.post('/post',data=dict(select_video=video_name))
    assert response.status_code == status_code
