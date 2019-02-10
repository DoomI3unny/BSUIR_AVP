#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <xmmintrin.h>

const int EXTERNAL_SIZE = 400;
const int INTERNAL_STRINGS = 4;
const int INTERNAL_COLOMNS = 8;

const int MILISECONDS = 1000;

const int UPPER_BOUND = 10000;
const int DELIMETER = 100;

float **CreateMatrix(const int external_size, const int internal_strings, const int internal_colomns)
{
	float **data = (float **)_aligned_malloc(external_size * external_size * sizeof(float *), 16);
	for (int i = 0; i < external_size * external_size; i++)
	{
		data[i] = (float *)_aligned_malloc(internal_strings * internal_colomns * sizeof(float), 16);
	}

	return data;
}

void FillMatrix(float **data, const int external_size, const int internal_strings, const int internal_colomns)
{
	srand((unsigned)time(NULL));
	
	for (int i = 0; i < external_size * external_size; i++)
	{
		for (int j = 0; j < internal_strings * internal_colomns; j++)
		{
			data[i][j] = (float)(rand() % UPPER_BOUND) / DELIMETER;			
		}
	}
}

void FillMatrixWithZeros(float **data, const int external_size, const int internal_strings, const int internal_colomns)
{
	for (int i = 0; i < external_size * external_size; i++)
	{
		for (int j = 0; j < internal_strings * internal_colomns; j++)
		{
			data[i][j] = 0;
		}
	}
}

void ShowMatrix(float **data, const int external_size, const int internal_strings, const int internal_colomns)
{
	for (int i = 0; i < external_size; i++)
	{
		for (int j = 0; j < external_size; j++)
		{
			printf("Matrix at [%d][%d]:\n", i, j);

			for (int m = 0; m < internal_strings; m++)
			{
				for (int n = 0; n < internal_colomns; n++)
				{
					printf("%16.2F", data[i * external_size + j][m * internal_colomns + n]);
				}
				printf("\n");
			}
			printf("------------------------------------\n");
		}
	}
}

void MatrixMultiply(float **a, float **b, float **c, const int external_size, const int internal_strings, const int internal_colomns)
{	
	for (int i = 0; i < external_size; i++)
	{
		for (int j = 0; j < external_size; j++)
		{
			for (int f = 0; f < external_size; f++)
			{
				float* __restrict temp_a = a[i * external_size + f];
				float* __restrict temp_b = b[f * external_size + j];
				float* __restrict temp_c = c[i * external_size + j];

				for (int m = 0; m < internal_strings; m++)
				{
					for (int n = 0; n < internal_colomns; n++)
					{
#pragma loop(no_vector)
						for (int l = 0; l < internal_strings; l++)
						{
							temp_c[m * internal_strings + l] += temp_a[m * internal_colomns + n] * temp_b[n * internal_strings + l];
						}
					}
				}
			}
		}
	}
}

void MatrixMultiplySSE(float **a, float **b, float **c, const int external_size, const int internal_strings, const int internal_colomns)
{
	for (int i = 0; i < external_size; i++)
	{
		for (int j = 0; j < external_size; j++)
		{
			for (int f = 0; f < external_size; f++)
			{
				float *temp_a = a[i * external_size + f];
				float *temp_b = b[f * external_size + j];
				float *temp_c = c[i * external_size + j];

				for (int m = 0; m < internal_strings; m++)
				{
					for (int n = 0; n < internal_colomns; n++)
					{
						__m128 sse_a = _mm_set1_ps(a[i * external_size + f][m * internal_colomns + n]);
						for (int l = 0; l < internal_strings; l += 4)
						{
							
							__m128 sse_b = _mm_load_ps(temp_b + n * internal_strings + l);
							__m128 sse_c = _mm_load_ps(temp_c + m * internal_strings + l);

							__m128 mul = _mm_mul_ps(sse_a, sse_b);							
							sse_c = _mm_add_ps(sse_c, mul);

							_mm_store_ps(temp_c + m * internal_strings + l, sse_c);
						}
					}
				}
			}
		}
	}
}

bool MatrixCompare(float **a, float **b, const int external_size, const int internal_strings, const int internal_colomns)
{
	for (int i = 0; i < external_size * external_size; i++)
	{
		for (int j = 0; j < internal_strings * internal_colomns; j++)
		{
			if (a[i][j] != b[i][j])
			{
				return false;
			}
		}
	}

	return true;
}

void DeleteMatrix(float **data, const int external_size, const int internal_strings, const int internal_colomns)
{
	for (int i = 0; i < external_size * external_size; i++)
	{
		_aligned_free(data[i]);
	}
	_aligned_free(data);
}

int main()
{
	DWORD start, end;

	float **A = CreateMatrix(EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_COLOMNS);
	FillMatrix(A, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_COLOMNS);	
	
	float **B = CreateMatrix(EXTERNAL_SIZE, INTERNAL_COLOMNS, INTERNAL_STRINGS);
	FillMatrix(B, EXTERNAL_SIZE, INTERNAL_COLOMNS, INTERNAL_STRINGS);	

	float **C = CreateMatrix(EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS);
	FillMatrixWithZeros(C, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS);

	float **D = CreateMatrix(EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS);
	FillMatrixWithZeros(D, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS);

	start = GetTickCount();	
	MatrixMultiply(A, B, C, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_COLOMNS);	
	end = GetTickCount();	
	

	printf("Mutiplication time: %.3lf\n", (double)(end - start) / MILISECONDS);	


	start = GetTickCount();
	MatrixMultiplySSE(A, B, D, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_COLOMNS);
	end = GetTickCount();

	
	printf("Multiplication using SSE: %0.3lf\n", (double)(end - start) / MILISECONDS);
	

	if (MatrixCompare(C, D, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS))
	{
		printf("\nMatrixes are equal");
	}
	else
	{
		printf("\nMatrixes aren't equal");
	}

	DeleteMatrix(A, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_COLOMNS);
	DeleteMatrix(B, EXTERNAL_SIZE, INTERNAL_COLOMNS, INTERNAL_STRINGS);
	DeleteMatrix(C, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS);
	DeleteMatrix(D, EXTERNAL_SIZE, INTERNAL_STRINGS, INTERNAL_STRINGS);

	_getch();
	return 0;
}