//#include <windows.h>
//LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);  // 窗口函数说明
//
//// 初始化窗口类
//// hInstance: 当前实例句柄，hPrevInst：父窗口句柄，lpszCmdLine：命令行，
//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInst, LPSTR lpszCmdLine,int nCmdShow)
//{
//	HWND hwnd;  // 窗口句柄
//	MSG Msg;  // 消息
//	WNDCLASS wndclass;  // wndclass类
//	char lpszClassName[] = "窗口";  // 类的名字定义为 窗口
//	char lpszTitle[] = "My_Windows";  // 窗口的标题名字
//
//	// 窗口类的定义
//	wndclass.style = 0;  // 窗口类型,缺省类型
//	wndclass.lpfnWndProc = WndProc;  // 窗口处理函数
//	wndclass.cbClsExtra = 0;  // 定义为0，没有扩展
//	wndclass.cbWndExtra = 0;  //
//	wndclass.hInstance = hInstance;
//	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
//	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);  // 窗口采用箭头光标
//	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);  // 窗口背景为白色
//	wndclass.lpszMenuName = NULL;
//	wndclass.lpszClassName = lpszClassName;
//
//	// 以下进行窗口类的注册
//	if (!RegisterClass(&wndclass))
//	{
//		MessageBeep(0);
//		return FALSE;
//	}
//
//	// 创建窗口
//	hwnd = CreateWindow(
//		lpszClassName,
//		lpszTitle,
//		WS_OVERLAPPEDWINDOW,
//		CW_USEDEFAULT,
//		CW_USEDEFAULT,
//		CW_USEDEFAULT,
//		CW_USEDEFAULT,
//		NULL,
//		NULL,
//		hInstance,
//		NULL
//		);
//
//	ShowWindow(hwnd, nCmdShow);  // 显示窗口
//	UpdateWindow(hwnd);  // 绘制用户区
//	while (GetMessage(&Msg, NULL, 0, 0))  // 消息循环
//	{
//		TranslateMessage(&Msg);
//		DispatchMessage(&Msg);
//	}
//	return Msg.wParam;
//
//}
//
//// hwnd:哪一个窗口发过来的消息，message：消息内容是什么，wParam、lParam：消息的附加值
//LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
//{
//	switch (message)
//	{
//		case WM_DESTROY:
//			PostQuitMessage(0);
//		default:
//			return DefWindowProc(hwnd, message, wParam, lParam);
//	}
//	return (0);		
//}