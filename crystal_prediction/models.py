from django.db import models

class CrystalStructure(models.Model):
    formula = models.CharField(max_length=100)
    spacegroup = models.CharField(max_length=50)
    formation_energy = models.FloatField()
    e_above_hull = models.FloatField()
    band_gap = models.FloatField()
    nsites = models.IntegerField()
    density = models.FloatField()
    volume = models.FloatField()
    has_bandstructure = models.BooleanField()

    def __str__(self):
        return self.formula
