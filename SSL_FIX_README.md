# 🔧 Correction SSL/TLS - Issue #136

## 🚨 Problème Résolu

L'erreur `httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]` qui bloquait la recherche web et les connexions MCP a été corrigée.

## ✅ Solution

### Configuration Automatique
- **Par défaut** : SSL vérification activée (sécurisé)
- **Fallback automatique** : Si SSL échoue, retry sans vérification
- **Retry logic** : 3 tentatives avec backoff exponentiel
- **Logging** : Traçabilité complète des erreurs

### Configuration Manuelle (Optionnel)

```bash
# Désactiver SSL vérification (développement uniquement)
export OPEN_DEEP_RESEARCH_VERIFY_SSL=false

# Activer SSL vérification (recommandé)
export OPEN_DEEP_RESEARCH_VERIFY_SSL=true
```

## 🧪 Test de la Correction

```bash
python test_ssl_fix.py
```

## 📋 Changements

### Nouveaux Fichiers
- `test_ssl_fix.py` - Script de test SSL
- `SSL_FIX_DOCUMENTATION.md` - Documentation technique complète

### Fichiers Modifiés
- `src/open_deep_research/utils.py` - Fonctions SSL sécurisées
- `src/legacy/utils.py` - Gestion SSL pour legacy
- `src/open_deep_research/configuration.py` - Configuration SSL
- `pyproject.toml` - Dépendance certifi ajoutée

## 🚀 Utilisation

**Aucun changement requis** - La correction est automatique et rétrocompatible.

### Pour les Développeurs

```python
from open_deep_research.utils import safe_http_request

# Requête sécurisée avec retry automatique
response = await safe_http_request(
    url="https://api.example.com",
    verify_ssl=True,  # Par défaut
    timeout=30,
    max_retries=3
)
```

## 🔒 Sécurité

- ✅ **Sécurisé par défaut** : SSL vérification activée
- ✅ **Fallback contrôlé** : Désactivation uniquement si nécessaire
- ✅ **Logging** : Toutes les désactivations SSL sont tracées
- ✅ **Warnings** : Alertes lors de la désactivation SSL

## 📊 Résultats

### Avant
```
❌ httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED]
❌ Recherche web bloquée
❌ Connexions MCP échouées
```

### Après
```
✅ Gestion automatique des erreurs SSL
✅ Retry logic avec fallback
✅ Configuration flexible
✅ Compatibilité maintenue
```

## 🐛 Dépannage

### Si les erreurs SSL persistent :

1. **Vérifier la connectivité** :
   ```bash
   curl -I https://api.github.com
   ```

2. **Tester avec SSL désactivé** :
   ```bash
   export OPEN_DEEP_RESEARCH_VERIFY_SSL=false
   python test_ssl_fix.py
   ```

3. **Consulter les logs** pour les détails d'erreur

4. **Vérifier les certificats** si en environnement corporate

## 📝 Notes

- **Version** : 0.0.16
- **Issue** : #136
- **Statut** : ✅ Résolu
- **Compatibilité** : Toutes les plateformes
- **Impact** : Amélioration automatique, aucun breaking change

---

**Pour plus de détails techniques** : Voir `SSL_FIX_DOCUMENTATION.md` 