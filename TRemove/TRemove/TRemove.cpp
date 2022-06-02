#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Windows.h>

#define ITER 125000000

typedef struct
{
    long int val;
} myStruct;

DWORD WINAPI work(LPVOID param)
{
    for (int j = 0; j < 10; j++)
    {
        for (int i = 2; i < ITER; i++)
        ((myStruct*)param)->val += i * i - i;
    }

    return 0;
}

int main()
{
    myStruct a = { 1 };
    myStruct b = { 1 };
    myStruct c = { 1 };

    clock_t start = clock();

    HANDLE t1 = CreateThread(NULL, 0, work, &a, 0, NULL);
    HANDLE t2 = CreateThread(NULL, 0, work, &b, 0, NULL);
    HANDLE t3 = CreateThread(NULL, 0, work, &c, 0, NULL);
    
    WaitForSingleObject(t1, INFINITE);
    WaitForSingleObject(t2, INFINITE);
    WaitForSingleObject(t3, INFINITE);

    clock_t end= clock();

    printf("%.3fs\n", (double)(end - start) / CLOCKS_PER_SEC);

    printf("%d | %d | %d", a.val, b.val, c.val);
}
