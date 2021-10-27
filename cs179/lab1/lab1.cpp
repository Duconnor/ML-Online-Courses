#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

void test1() {
    int *a = NULL;
    a = (int*) malloc(sizeof(int));
    if (a != NULL) {
        *a = 3;
        *a = *a + 2;
        printf("%d\n", *a);
        free(a);
    }
}

void test2() {
    int *a, *b;
    a = (int *) malloc(sizeof (int));
    b = (int *) malloc(sizeof (int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}

void test3() {
    int i, *a = (int *) malloc(1000 * sizeof(int));

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i + a) = i;
}

void test4() {
    int **a = (int **) malloc(3 * sizeof (int *));
    for (int i = 0; i < 3; i++) {
        a[i] = (int*)malloc(100 * sizeof(int *));
    }
    a[1][1] = 5;
}

void test5() {
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (!*a)
        printf("Value is 0\n");
}

int main() {
    test5();
}