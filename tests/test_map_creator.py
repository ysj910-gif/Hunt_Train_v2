# tests/test_map_creator.py
import sys
import os
import unittest

# 상위 디렉토리를 경로에 추가하여 모듈 import 가능하게 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.map_creator import MapCreator

class TestMapCreator(unittest.TestCase):
    def setUp(self):
        # Mock Agent 없이도 수동 주입 기능으로 테스트 가능
        self.creator = MapCreator(agent=None)

    def test_add_platform_logic(self):
        """정상적인 발판 추가 테스트"""
        print("\n[Test] Platform Logic")
        
        # 1. 시작점 설정 (수동 주입)
        self.creator.set_manual_pos(100, 500)
        success, pos = self.creator.set_start_point()
        self.assertTrue(success)
        self.assertEqual(pos, (100, 500))

        # 2. 종료점 설정
        self.creator.set_manual_pos(300, 504) # Y값 약간 차이 둠 (평균값 테스트)
        success, pos = self.creator.set_end_point()
        self.assertTrue(success)
        
        # 3. 발판 추가
        success, plat = self.creator.add_platform()
        self.assertTrue(success)
        
        # 검증
        self.assertEqual(plat['x_start'], 100)
        self.assertEqual(plat['x_end'], 300)
        self.assertEqual(plat['y'], 502) # (500+504)/2
        print(f"-> Generated Platform: {plat}")

    def test_invalid_platform(self):
        """너무 짧은 발판 방지 테스트"""
        print("\n[Test] Invalid Platform")
        self.creator.set_manual_pos(100, 500)
        self.creator.set_start_point()
        self.creator.set_manual_pos(102, 500) # 길이 2
        self.creator.set_end_point()
        
        success, msg = self.creator.add_platform()
        self.assertFalse(success)
        print(f"-> Expected Failure: {msg}")

    def test_undo(self):
        """실행 취소 테스트"""
        print("\n[Test] Undo Logic")
        # 발판 하나 추가
        self.creator.set_manual_pos(10, 10)
        self.creator.set_start_point()
        self.creator.set_manual_pos(50, 10)
        self.creator.set_end_point()
        self.creator.add_platform()
        
        self.assertEqual(len(self.creator.new_platforms), 1)
        
        # 취소 실행
        self.creator.undo_last_platform()
        self.assertEqual(len(self.creator.new_platforms), 0)
        print("-> Undo Successful")

if __name__ == '__main__':
    unittest.main()