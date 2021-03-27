# Classificação de lesões de pele utilizando redes neurais convolucionais: Um guia educacional
 
 O objetivo deste guia é apresentar um passo a passo para as pessoas que tenham interesse em trabalhar ou entender melhor o problema de classificação de lesões de pele. Independente do nível de conhecimento sobre aprendizagem de máquina, programação e conceitos de medicina.
 
 Link do Notebook: https://nbviewer.jupyter.org/github/felipezago/skin-cancer-classification/blob/main/jupyter/Classifica%C3%A7%C3%A3o%20de%20Les%C3%B5es%20de%20Pele.ipynb
 
# Procedimentos de Instalação
Iniciaremos com a instalação do ambiente e das ferramentas necessárias para a execução das atividades.

Como ambiente utilizamos o Anaconda Navigator.
Como IDE utilizaremos o Spyder, mas caso desejado pode ser utilizado qualquer outra.
Link para download do Anaconda: https://www.anaconda.com/products/individual#Downloads

Após instalar, deve ser criado um ambiente através do Anaconda Navigator e, nesse ambiente, instalar o "CMD Prompt" que estará disponível na tela principal do ambiente.

# Ferramentas
Agora, instalaremos as ferramentas necessárias para execução das atividades. Para cada uma das ferramentas será apresentado um breve resumo de suas aplicações e funcionalidades.

Pytorch e relacionados
PyTorch é uma biblioteca de aprendizado de máquina de código aberto baseada na biblioteca Torch. Trata-se de uma biblioteca muito utilizada em aplicativos que utilizam visão computacional e processamento de linguagem natural. A biblioteca surgiu e foi apresentada após o desenvolvimento de alguns projetos no laboratório de pesquisa de IA do Facebook.

Torchvision
A bibilioteca Torchvision é parte do projeto Pytorch. Torchvision é uma biblioteca criada para trabalhar com visão computacional.

Para mais informações, acesse: https://pytorch.org/vision/master/

CUDA Toolkit
O CUDA Toolkit fornece um ambiente de desenvolvimento para a criação de aplicativos acelerados por GPU (Graphics processing unit) ou unidade de processamento gráfico. Trata-se de uma alternativa para usuários que possuem placa de vídeo com suporte a GPU.

Para mais informações, acesse: https://developer.nvidia.com/cuda-toolkit

Para instalação desses pacotes deve ser utilizado o seguinte comando no terminal do ambiente criado no Anaconda:

# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

Pandas
Pandas é um pacote Python que fornece importantes ferramentas de análise de dados e estruturas de dados de alta performance fáceis de usar.

Para mais informações, acesse: https://pandas.pydata.org/

Para instalação do Pandas deve ser utilizado o seguinte comando no terminal do ambiente criado no Anaconda:

# conda install pandas

MatPlotLib 
Matplotlib é uma biblioteca que permite a visualização animada ou interativa de dados estatísticos, em Python. Nesse trabalho, a biblioteca será utilizada para criação e apresentação de gráficos que podem facilitar o entendimento de alguns resultados obtidos.

Para mais informações, acesse: https://matplotlib.org/stable/index.html

Para instalação do Matplotlib deve ser utilizado o seguinte comando no terminal do ambiente criado no Anaconda:

# conda install matplotlib

Scikit Learn
Scikit Learn ou sklearn é uma biblioteca para a linguagem de programação python utilizada para aplicações de aprendizado de máquina. Ela dispõe de várias ferramentas para implementação de vários algoritmos de classificação, inclusive os que foram utilizados no nosso trabalho.

Trata-se de uma excelente opção para iniciar os trabalhos e estudos em aprendizado de máquina.

Para mais informações, acesse: http://scikit-learn.org/

Para instalação do Sklearn deve ser utilizado o seguinte comando no terminal do ambiente criado no Anaconda:

# conda install scikit-learn

Seaborn
Seaborn é uma biblioteca de visualização de dados em Python baseada em Matplotlib. Ela fornece uma interface de alto nível para criação de gráficos que seriam de manipulação complexa utilizando a biblioteca Matplotlib na criação de informações estatísticas atraentes e informativas.

Para mais informações, acesse: https://seaborn.pydata.org/

Para instalação do Seaborn deve ser utilizado o seguinte comando no terminal do ambiente criado no Anaconda:

# conda install seaborn

ImageIO
ImageIO é uma biblioteca Python que fornece uma interface fácil para ler e gravar uma ampla variedade de dados de imagem, incluindo imagens animadas e formatos científicos.

Para mais informações, acesse: https://imageio.github.io

Para instalação do ImageIO deve ser utilizado o seguinte comando no terminal do ambiente criado no Anaconda:

# conda install imageio
 
 
