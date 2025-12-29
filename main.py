#!/usr/bin/env python3
"""
杂草识别与分析系统主程序入口
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow

def main():
    """
    主函数
    """
    try:
        # 创建并运行主窗口
        app = MainWindow()
        app.run()
    except Exception as e:
        print(f"运行错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()