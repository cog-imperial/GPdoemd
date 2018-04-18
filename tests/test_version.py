
import pytest
import GPdoemd

class TestVersion:

	def test_version(self):
		assert isinstance( GPdoemd.__version__, str )

	def test_author(self):
		assert GPdoemd.__author__  == 'Simon Olofsson'

	def test_license (self):
		assert GPdoemd.__license__ == 'MIT License'
