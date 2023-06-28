//#include<windows.h>
//#include<stdlib.h>
//#include<string.h>
//
//long WINAPI WndProc(HWND hWnd, UINT iMessage, UINT wParam, LONG lParam);
//BOOL InitWindowsClass(HINSTANCE hInstance);
//BOOL InitWindows(HINSTANCE hInstance, int nCmdShow);
//
//// ������
//int WINAPI WinMain(HINSTANCE hInstance,
//	HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)  
//{
//	MSG Message;
//	if (!InitWindowsClass(hInstance)) return FALSE;  // ������ĳ�ʼ��
//	if (!InitWindows(hInstance, nCmdShow)) return FALSE;  // ���ڳ�ʼ��
//	while (GetMessage(&Message, 0, 0, 0))  // ��Ϣѭ��
//	{
//		TranslateMessage(&Message);
//		DispatchMessage(&Message);
//	}
//	return Message.wParam;
//}
//long WINAPI WndProc(HWND hWnd, UINT iMessage, UINT wParam, LONG lParam)  // ��������Ϣ
//{
//	HDC hDC;  // �����豸�������
//	HBRUSH hBrush;  // ���廭ˢ�ľ��
//	HPEN hPen;  // ���廭�ʵľ��
//	PAINTSTRUCT PtStr;  // ����ָ�������ͼ��Ϣ�Ľṹ�����
//	// ����һ��POINT���飬 ����6����
//	POINT points[6] = { {100, 212}, {70, 227}, {70, 250}, {130, 250},{130, 227}, {100, 212} };  // ��һ��������һ����Ҫһ�����պ�ͼ��
//	switch (iMessage)  // ������Ϣ��paint��Ϣ
//	{
//		case WM_PAINT:  // �����ͼ��Ϣ
//			hDC = BeginPaint(hWnd, &PtStr);  // ����DC
//			hPen = (HPEN)GetStockObject(NULL_PEN);  // ��ȡϵͳ����Ŀջ��ʡ��ձʣ��ѱ��Ƶ���ʼλ��
//			SelectObject(hDC, hPen);  // ѡ�񻭱�
//			hBrush = (HBRUSH)GetStockObject(BLACK_BRUSH);  // ��ȡϵͳ����Ļ�ˢ
//			SelectObject(hDC, hBrush);  // ѡ��ˢ
//			LineTo(hDC, 50, 50);  // ����
//			DeleteObject(hPen);  // ɾ������
//			hPen = CreatePen(PS_SOLID, 2, RGB(255, 0, 0));  // ��������
//			SelectObject(hDC, hPen);  // ѡ�񻭱�
//			// ��һ��������
//			LineTo(hDC, 150, 50);
//			LineTo(hDC, 100, 137);
//			LineTo(hDC, 50, 50);
//			Polyline(hDC, points, 6);  // ��һ�������
//			Arc(hDC, 63, 137, 138, 212, 100, 137, 100, 137);  // ��һ��Բ
//			Pie(hDC, 213, 137, 288, 212, 240, 137, 260, 137);  // ��һ��Բ��
//			Rectangle(hDC, 213, 212, 287, 250);  // ��һ��������
//			RoundRect(hDC, 213, 100, 287, 137, 20, 20);  // ��һ��Բ�ǳ�����
//			DeleteObject(hPen);  // ɾ������
//			DeleteObject(hBrush);  // ɾ����ˢ
//			EndPaint(hWnd, &PtStr);  // ������ͼ
//			return 0;
//		case WM_DESTROY:  // ����Ӧ�ó���
//			PostQuitMessage(0);
//			return 0;
//		default:  // ������Ϣ�������
//			return (DefWindowProc(hWnd, iMessage, wParam, lParam));
//	}
//}
//BOOL InitWindows(HINSTANCE hInstance, int nCmdShow)  // ��ʼ������
//{
//	HWND hWnd;  // Ҫ����һ�����ڣ���Ҫ���崰�ھ��
//	hWnd = CreateWindow("WinFill",  // ���ɴ���
//		"���ʾ������",
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
//	ShowWindow(hWnd, nCmdShow);  // ��ʾ����
//	UpdateWindow(hWnd);
//	return TRUE;
//}
//BOOL InitWindowsClass(HINSTANCE hInstance)  // ���崰����
//{
//	WNDCLASS WndClass;
//	WndClass.cbClsExtra = 0;
//	WndClass.cbWndExtra = 0;
//	WndClass.hbrBackground = (HBRUSH)(GetStockObject(WHITE_BRUSH));  // ��ɫ��ˢ
//	WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
//	WndClass.hIcon = LoadIcon(NULL, "END");
//	WndClass.hInstance = hInstance;
//	WndClass.lpfnWndProc = WndProc;  // ��Ϣ��Ĵ�����
//	WndClass.lpszClassName = "WinFill";  // ���ʼ���ĵ�һ��������Ӧ����
//	WndClass.lpszMenuName = NULL;
//	WndClass.style = CS_HREDRAW | CS_VREDRAW;
//	return RegisterClass(&WndClass);
//}
