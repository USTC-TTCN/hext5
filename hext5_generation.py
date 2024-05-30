'''
@usage: python this_file.py
@desc: use trained model to predict summary and identifier
'''
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

max_input_length = 4096
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

# predict summary
def predict_summary(model,tokenizer,code):
    input = tokenizer('summarize: '+code,return_tensors='pt',max_length=max_input_length,truncation=True).to(device)
    output = model.generate(**input,max_new_tokens=256)[0]
    return tokenizer.decode(output,skip_special_tokens=True)


# predict identifier (func name)
def predict_identifier(model,tokenizer,code):
    '''
    code should be like: "unsigned __int8 *__cdecl <func>(int *<var_0>,...){ return <func_1>(1);}"
    '''
    input = tokenizer('identifier_predict: '+code,return_tensors='pt',max_length=max_input_length,truncation=True).to(device)
    output = model.generate(**input)[0]
    return tokenizer.decode(output)


if __name__ == '__main__':
    model_path = "./checkpoint"
    model = T5ForConditionalGeneration.from_pretrained(model_path, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    code = "void __fastcall __noreturn <func>(int a1, int a2, char a3, __int64 a4, __int64 a5)\n{\n  __int64 v5; // rdi\n  int v6; // ebx\n  const char *v9; // rsi\n  char *v10; // r12\n  char *v11; // r13\n  char *v12; // rax\n  char v13[42]; // [rsp+Eh] [rbp-2Ah] BYREF\n\n  v5 = (unsigned int)(a1 - 1);\n  v6 = status;\n  if ( (unsigned int)v5 <= 3 )\n  {\n    v9 = (&off_413A60)[v5];\n    if ( a2 < 0 )\n    {\n      v13[0] = a3;\n      v11 = v13;\n      v10 = &asc_412691[-a2];\n      v13[1] = 0;\n    }\n    else\n    {\n      v10 = \"--\";\n      v11 = *(char **)(a4 + 32LL * a2);\n    }\n    v12 = dcgettext(0LL, v9, 5);\n    error(v6, 0, v12, v10, v11, a5);\n    abort();\n  }\n  abort();\n}\n"
    
    res = predict_summary(model,tokenizer,code)
    print("[+] summarization:\n",res)
    res = predict_identifier(model,tokenizer,code)
    print("[+] identifier prediction:\n",res)
