//#include <windows.h>
//LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);  // ���ں���˵��
//
//// ��ʼ��������
//// hInstance: ��ǰʵ�������hPrevInst�������ھ����lpszCmdLine�������У�
//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInst, LPSTR lpszCmdLine,int nCmdShow)
//{
//	HWND hwnd;  // ���ھ��
//	MSG Msg;  // ��Ϣ
//	WNDCLASS wndclass;  // wndclass��
//	char lpszClassName[] = "����";  // ������ֶ���Ϊ ����
//	char lpszTitle[] = "My_Windows";  // ���ڵı�������
//
//	// ������Ķ���
//	wndclass.style = 0;  // ��������,ȱʡ����
//	wndclass.lpfnWndProc = WndProc;  // ���ڴ�����
//	wndclass.cbClsExtra = 0;  // ����Ϊ0��û����չ
//	wndclass.cbWndExtra = 0;  //
//	wndclass.hInstance = hInstance;
//	wndclass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
//	wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);  // ���ڲ��ü�ͷ���
//	wndclass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);  // ���ڱ���Ϊ��ɫ
//	wndclass.lpszMenuName = NULL;
//	wndclass.lpszClassName = lpszClassName;
//
//	// ���½��д������ע��
//	if (!RegisterClass(&wndclass))
//	{
//		MessageBeep(0);
//		return FALSE;
//	}
//
//	// ��������
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
//	ShowWindow(hwnd, nCmdShow);  // ��ʾ����
//	UpdateWindow(hwnd);  // �����û���
//	while (GetMessage(&Msg, NULL, 0, 0))  // ��Ϣѭ��
//	{
//		TranslateMessage(&Msg);
//		DispatchMessage(&Msg);
//	}
//	return Msg.wParam;
//
//}
//
//// hwnd:��һ�����ڷ���������Ϣ��message����Ϣ������ʲô��wParam��lParam����Ϣ�ĸ���ֵ
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