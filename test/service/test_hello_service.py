import unittest

from app.service.hello_service import HelloService


class TestHelloService(unittest.TestCase):

    def test_get_hello(self):
        hello_service = HelloService()
        self.assertEqual(hello_service.get_hello(), {"hello": "world"})

