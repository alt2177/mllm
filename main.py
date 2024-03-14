import mllm

def main():
	my_mllm = mllm.MLLM()
	my_mllm.train()
	my_mllm.write_results()
	print("DONE!")


if __name__ == "__main__":
	main()
