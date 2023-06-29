//#include<windows.h>
//#include<stdlib.h>
//#include<string.h>
//
//long WINAPI WndProc(HWND hWnd, UINT iMessage, UINT wParam, LONG lParam);
//BOOL InitWindowsClass(HINSTANCE hInstance);
//BOOL InitWindows(HINSTANCE hInstance, int nCmdShow);
//
//// 主函数
//int WINAPI WinMain(HINSTANCE hInstance,
//	HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)  
//{
//	MSG Message;
//	if (!InitWindowsClass(hInstance)) return FALSE;  // 窗口类的初始化
//	if (!InitWindows(hInstance, nCmdShow)) return FALSE;  // 窗口初始化
//	while (GetMessage(&Message, 0, 0, 0))  // 消息循环
//	{
//		TranslateMessage(&Message);
//		DispatchMessage(&Message);
//	}
//	return Message.wParam;
//}
//long WINAPI WndProc(HWND hWnd, UINT iMessage, UINT wParam, LONG lParam)  // 处理窗口消息
//{
//	HDC hDC;  // 定义设备环境句柄
//	HBRUSH hBrush;  // 定义画刷的句柄
//	HPEN hPen;  // 定义画笔的句柄
//	PAINTSTRUCT PtStr;  // 定义指向包含绘图信息的结构体变量
//	// 定义一个POINT数组， 包括6个点
//	POINT points[6] = { {100, 212}, {70, 227}, {70, 250}, {130, 250},{130, 227}, {100, 212} };  // 第一个点和最后一个点要一样，闭合图形
//	switch (iMessage)  // 处理消息，paint消息
//	{
//		case WM_PAINT:  // 处理绘图消息
//			hDC = BeginPaint(hWnd, &PtStr);  // 创建DC
//			hPen = (HPEN)GetStockObject(NULL_PEN);  // 获取系统定义的空画笔。空笔：把笔移到起始位置
//			SelectObject(hDC, hPen);  // 选择画笔
//			hBrush = (HBRUSH)GetStockObject(BLACK_BRUSH);  // 获取系统定义的画刷
//			SelectObject(hDC, hBrush);  // 选择画刷
//			LineTo(hDC, 50, 50);  // 画线
//			DeleteObject(hPen);  // 删除画笔
//			hPen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));  // 创建画笔
//			SelectObject(hDC, hPen);  // 选择画笔
//			// 画一个三角形
//			LineTo(hDC, 150, 50);
//			LineTo(hDC, 100, 137);
//			LineTo(hDC, 50, 50);
//			Polyline(hDC, points, 6);  // 画一个五边形
//			Arc(hDC, 63, 137, 138, 212, 100, 137, 100, 137);  // 画一个圆
//			Pie(hDC, 213, 137, 288, 212, 240, 137, 260, 137);  // 画一个圆饼
//			Rectangle(hDC, 213, 212, 287, 250);  // 画一个长方形
//			RoundRect(hDC, 213, 100, 287, 137, 20, 20);  // 画一个圆角长方形
//			DeleteObject(hPen);  // 删除画笔
//			DeleteObject(hBrush);  // 删除画刷
//			EndPaint(hWnd, &PtStr);  // 结束绘图
//			return 0;
//		case WM_DESTROY:  // 结束应用程序
//			PostQuitMessage(0);
//			return 0;
//		default:  // 其他消息处理程序
//			return (DefWindowProc(hWnd, iMessage, wParam, lParam));
//	}
//}
//BOOL InitWindows(HINSTANCE hInstance, int nCmdShow)  // 初始化窗口
//{
//	HWND hWnd;  // 要创建一个窗口，就要定义窗口句柄
//	hWnd = CreateWindow("WinFill",  // 生成窗口
//		"填充示例程序",
//		WS_OVERLAPPEDWINDOW,
//		CW_USEDEFAULT,
//		0,
//		CW_USEDEFAULT,
//		0,
//		NULL,
//		NULL,
//		hInstance,
//		NULL);
//	if (!hWnd)
//		return FALSE;
//	ShowWindow(hWnd, nCmdShow);  // 显示窗口
//	UpdateWindow(hWnd);
//	return TRUE;
//}
//BOOL InitWindowsClass(HINSTANCE hInstance)  // 定义窗口类
//{
//	WNDCLASS WndClass;
//	WndClass.cbClsExtra = 0;
//	WndClass.cbWndExtra = 0;
//	WndClass.hbrBackground = (HBRUSH)(GetStockObject(WHITE_BRUSH));  // 白色画刷
//	WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
//	WndClass.hIcon = LoadIcon(NULL, "END");
//	WndClass.hInstance = hInstance;
//	WndClass.lpfnWndProc = WndProc;  // 消息类的处理函数
//	WndClass.lpszClassName = "WinFill";  // 与初始化的第一个参数对应起来
//	WndClass.lpszMenuName = NULL;
//	WndClass.style = CS_HREDRAW | CS_VREDRAW;
//	return RegisterClass(&WndClass);
//}
