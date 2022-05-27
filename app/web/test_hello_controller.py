import unittest
from unittest.mock import MagicMock

from app.service.hello_service import HelloService
from app.web.hello_controller import HelloController


class TestHelloController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mocked_hello_service = HelloService()
        mocked_hello_service.get_hello = MagicMock(return_value={"mocked": "value"})
        cls._hello_Controller = HelloController(mocked_hello_service)

    @classmethod
    def tearDownClass(cls):
        print('test down')

    def test_hello(self):
        hello = self._hello_Controller.hello()
        self.assertEqual(hello, {"mocked": "value"})
