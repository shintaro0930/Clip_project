# How To Run

You should use GPU taht is called CUDA (version is anything ok).

<br><br>


## 1. You need to take in a lot of pictures to dir `/pictures/`
and then, you may have the pictures that include punctuation '.HEIC' or '.heic', so you have to change your pictures punctuations.

So, Run the code.
```zsh:change_extension.py
python3 change_extension.py
```

In this file, your all pictures puctuations in '/pictures/' will be changed '.JPG'.

python cannot read the extension '.HEIC' or '.heic'.

<br><br>

## 2. run the code1
run the code that changes image to texts and probabilities.

```zsh:clip.py
python3 clip.py
```

then, your vague pictures will be setted at `/used_picntures/`

and, your result will be setted at `/output.txt`.

<br><br>

## 3. run the code2
run the code that calucurate the probabilities between your input text and the image(changed img2txt) with cosine simularity.

```zsh:get_ouyput.py
python3 get_output.py
```