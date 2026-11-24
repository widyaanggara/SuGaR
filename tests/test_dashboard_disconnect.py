from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_dashboard.server import DashboardHandler


class DisconnectTest(unittest.TestCase):
    def handler(self):
        handler = object.__new__(DashboardHandler)
        handler.close_connection = False
        handler.server = SimpleNamespace()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.wfile = Mock()
        return handler

    def test_disconnect_during_headers_or_payload_is_not_a_disk_error(self):
        for error in (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            for phase in ('headers', 'body'):
                with self.subTest(error=error, phase=phase):
                    handler = self.handler()
                    failing = handler.end_headers if phase == 'headers' else handler.wfile.write
                    failing.side_effect = error()
                    handler.send_json({'images': []})
                    self.assertTrue(handler.close_connection)
                    handler.send_response.assert_called_once_with(200)

    def test_real_io_errors_are_not_silenced(self):
        handler = self.handler()
        handler.wfile.write.side_effect = OSError('unrelated I/O error')
        with self.assertRaises(OSError):
            handler.send_json({})

    def test_get_file_disconnect_does_not_send_second_response(self):
        handler = self.handler()
        handler.path = '/'
        handler.send_file = Mock(side_effect=BrokenPipeError())
        handler.send_json = Mock()
        handler.do_GET()
        handler.send_json.assert_not_called()
        self.assertTrue(handler.close_connection)

    def test_post_disconnect_does_not_send_second_response(self):
        handler = self.handler()
        handler.path = '/api/unknown'
        handler.headers = {'Content-Length': '2'}
        handler.rfile = Mock()
        handler.rfile.read.side_effect = ConnectionResetError()
        handler.send_json = Mock()
        handler.do_POST()
        handler.send_json.assert_not_called()
        self.assertTrue(handler.close_connection)


if __name__ == '__main__':
    unittest.main()
