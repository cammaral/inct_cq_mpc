from pathlib import Path
import shutil

arquivo = Path("mpc/models.py")
codigo = arquivo.read_text()

antigo = '''                # O i-esimo circuito SEM entrada usa EXATAMENTE i qubits.
                self.aux = nn.ModuleList([
                    Circuit(i, int(cfg.get("layers", 2)), [word],
                            bool(cfg.get("quantum_trainable", True)))
                    for i, word in enumerate(words, start=1)
                ])'''

novo = '''                # Um unico circuito compartilhado por todos os termos.
                qubits = int(cfg.get("qubits", self.terms))
                self.aux = Circuit(
                    qubits,
                    int(cfg.get("layers", 2)),
                    words,
                    bool(cfg.get("quantum_trainable", True))
                )'''

antigo_coef = '''            return torch.cat([circuit().flatten() for circuit in self.aux])'''

novo_coef = '''            return self.aux().flatten()'''

if codigo.count(antigo) != 1 or codigo.count(antigo_coef) != 1:
    raise SystemExit(
        "Codigo diferente do esperado. Nenhum arquivo foi alterado."
    )

shutil.copy2(arquivo, "mpc/models_backup.py")

codigo = codigo.replace(antigo, novo)
codigo = codigo.replace(antigo_coef, novo_coef)

arquivo.write_text(codigo)

print("MPC atualizada: circuito unico compartilhado.")
print("Backup: mpc/models_backup.py")
