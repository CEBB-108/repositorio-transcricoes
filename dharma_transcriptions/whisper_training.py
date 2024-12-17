import os
import torch
import whisper
from whisper.tokenizer import get_tokenizer

# Caminho para salvar o modelo treinado
TRAINED_MODEL_PATH = os.path.join("trained_models", "whisper_finetuned.pt")

# Caminho dos dados de treinamento
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "TREINAMENTO"))
BRUTOS_PATH = os.path.join(BASE_PATH, "dados_brutos")
CORRIGIDOS_PATH = os.path.join(BASE_PATH, "referencia_corrigida")


def load_base_model():
    """
    Carrega o modelo base do Whisper e habilita o cálculo de gradientes.
    """
    print("[INFO] Carregando o modelo base Whisper...")
    model = whisper.load_model("base")  # Alterar para 'small', 'medium', etc., se necessário.

    # Habilitar cálculo de gradientes para ajuste fino
    for param in model.parameters():
        param.requires_grad = True

    print("[INFO] Modelo base Whisper carregado com sucesso.")
    return model


def fine_tune_model(model, brutos_dir, corrigidos_dir):
    """
    Realiza o fine-tuning do modelo Whisper usando os dados fornecidos.
    """
    print("[INFO] Preparando os dados para o treinamento...")

    # Configuração do otimizador e perda
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Taxa de aprendizado baixa
    loss_function = torch.nn.CrossEntropyLoss()

    # Tokenizer para codificar textos corrigidos
    tokenizer = get_tokenizer(multilingual=True)

    # Listar pares de arquivos para treinamento
    training_pairs = []
    for bruto_file in os.listdir(brutos_dir):
        if not bruto_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(bruto_file)[0]
        corrigido_file = os.path.join(corrigidos_dir, f"{base_name}.txt")

        if os.path.exists(corrigido_file):
            training_pairs.append((os.path.join(brutos_dir, bruto_file), corrigido_file))

    if not training_pairs:
        raise ValueError("[ERRO] Nenhum par válido encontrado para treinamento.")

    print(f"[INFO] Total de pares para treinamento: {len(training_pairs)}")

    # Loop de treinamento
    for epoch in range(1):  # Apenas 1 época para demonstração
        print(f"[INFO] Época {epoch + 1}")
        for bruto_path, corrigido_path in training_pairs:
            print(f"[DEBUG] Treinando com: {bruto_path} e {corrigido_path}")

            # Processar áudio e texto
            try:
                with open(bruto_path, "r", encoding="utf-8") as bruto_file, open(corrigido_path, "r", encoding="utf-8") as corrigido_file:
                    bruto_text = bruto_file.read()
                    corrigido_text = corrigido_file.read()

                # Carregar áudio bruto como mel-espectrograma
                audio_tensor = whisper.log_mel_spectrogram(torch.tensor([float(x) for x in bruto_text.split()]))

                # Codificar texto corrigido
                target_tokens = tokenizer.encode(corrigido_text)

                # Ajustar formatos para entrada e saída
                inputs = audio_tensor.unsqueeze(0)  # Adicionar dimensão batch
                targets = torch.tensor(target_tokens).unsqueeze(0)

                # Prever saída
                outputs = model.decoder(inputs, targets[:, :-1])

                # Calcular perda
                loss = loss_function(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"[INFO] Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"[ERRO] Falha durante o treinamento para {bruto_path}: {e}")

    print("[INFO] Fine-tuning concluído.")
    return model


def save_finetuned_model(model):
    """
    Salva o modelo treinado no caminho especificado.
    """
    os.makedirs(os.path.dirname(TRAINED_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)
    print(f"[INFO] Modelo treinado salvo em: {TRAINED_MODEL_PATH}")


if __name__ == "__main__":
    try:
        # Carregar o modelo base
        model = load_base_model()

        # Realizar o fine-tuning
        finetuned_model = fine_tune_model(model, BRUTOS_PATH, CORRIGIDOS_PATH)

        # Salvar o modelo treinado
        save_finetuned_model(finetuned_model)

        print("[INFO] Treinamento finalizado com sucesso!")
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro durante o treinamento: {e}")
