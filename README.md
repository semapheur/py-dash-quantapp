# Todo

- Implement sqlc

# Tips
Clone this repository with 

```
git clone https://github.com/semapheur/py-dash-quantapp
```

## Virtual environment

### venv
Create a Python virtual environment with

```
python -m venv ./<name>
```

Activate the virtual environment using

```
# Windows
<name>\scripts\activate.ps1
```

### Conda

```
conda create --name <env> --file <this file>
```

## Libraries
Install required Python libraries with

```
pip install -r requirements.txt
```

## TailwindCSS
Install TailwindCSS with

```
# Node
npm install
```

Build the CSS file with

```
npm tw-build
```

## Application

Run the application with

```
python app.py
```

# XBRL taxonomy
https://xbrl.us/home/filers/sec-reporting/taxonomies/


# Accounting terms

## Operating/Nonoperating/Noninterest Income/Expense

- NonoperatingGainsLosses = GainLossOnInvestments + [VentureCapitalGainsLossesNet] + DisposalGroupNotDiscontinuedOperationGainLossOnDisposal + [GainLossOnSaleOfStockInSubsidiaryOrEquityMethodInvestee] + DeconsolidationGainOrLossAmount + [GainLossOnSaleOfPreviouslyUnissuedStockBySubsidiaryOrEquityInvesteeNonoperatingIncome] + GainLossOnSaleOfInterestInProjects + [GainLossOnDerivativeInstrumentsNetPretax] + BusinessCombinationBargainPurchaseGainRecognizedAmount + OtherNonoperatingGainsLosses
- InvestmentIncomeNonoperating = NonoperatingGainsLosses + RoyaltyIncomeNonoperating + RentalIncomeNonoperating + DevelopmentProfitsNonoperating - RecoveryStrandedCosts + LeveragedLeasesIncomeStatementNetIncomeFromLeveragedLeases + InvestmentIncomeNet
- OperatingExpenses = FloorBrokerageExchangeAndClearanceFees + MarketingAndAdvertisingExpense + DepreciationAndAmortization + AmortizationOfMortgageServicingRightsMSRs + RestructuringSettlementAndImpairmentProvisions + OtherNonrecurringIncomeExpense + OtherExpenses + RoyaltyExpense + AccretionExpenseIncludingAssetRetirementObligations + PreOpeningCosts + LegalFees - GainLossOnRepurchaseOfDebtInstrument + InducedConversionOfConvertibleDebtExpense + GeneralAndAdministrativeExpense - [GainsLossesOnSalesOfAssets] - [GainLossOnSaleOfLeasedAssetsNetOperatingLeases] - [GainsLossesOnSalesOfOtherRealEstate] - [GainLossOnSaleOfStockInSubsidiaryOrEquityMethodInvestee] - [GainLossOnSaleOfPropertyPlantEquipment] - [GainLossOnSaleOfBusiness] - [GainsLossesOnExtinguishmentOfDebt] - [GainLossOnDerivativeInstrumentsNetPretax] + BusinessCombinationIntegrationRelatedCosts
- NonoperatingIncomeExpense = InvestmentIncomeNonoperating + GainLossOnContractTermination + GainLossOnCondemnation - LossFromCatastrophes + PublicUtilitiesAllowanceForFundsUsedDuringConstructionAdditions - SalesTypeLeaseInitialDirectCostExpenseCommencement - OperatingLeaseInitialDirectCostExpenseOverTerm + PublicUtilitiesAllowanceForFundsUsedDuringConstructionCapitalizedCostOfEquity - NetPeriodicDefinedBenefitsExpenseReversalOfExpenseExcludingServiceCostComponent + GovernmentAssistanceNonoperatingIncome + GovernmentAssistanceNonoperatingExpense + OtherNonoperatingIncomeExpense - UnusualOrInfrequentItemNetGainLoss + [ForeignCurrencyTransactionGainLossBeforeTax] + [GainLossOnSaleOfLeasedAssetsNetOperatingLeases] + [GainsLossesOnSalesOfOtherRealEstate] + [BankOwnedLifeInsuranceIncome] + [RealEstateInvestmentPartnershipRevenue] + [ConversionGainsAndLossesOnForeignInvestments] + [ProfitLossFromRealEstateOperations] - [MortgageServicingRightsMSRImpairmentRecovery] + [DebtInstrumentConvertibleBeneficialConversionFeature]
- NoninterestIncome = DividendIncomeOperating + InvestmentBankingAdvisoryBrokerageAndUnderwritingFeesAndCommissions + PrincipalTransactionsRevenue + PremiumsEarnedNet + NoninterestIncomeOtherOperatingIncome + NoninterestIncomeOther + [GainsLossesOnSalesOfAssets] + [GainLossOnSaleOfLeasedAssetsNetOperatingLeases] + [GainLossOnSaleOfStockInSubsidiaryOrEquityMethodInvestee] + [GainLossOnSaleOfPropertyPlantEquipment] + [GainLossOnSaleOfBusiness] + [GainLossOnDerivativeInstrumentsNetPretax] +  + [VentureCapitalGainsLossesNet] + [BankOwnedLifeInsuranceIncome] + [ProfitLossFromRealEstateOperations] + [RealEstateInvestmentPartnershipRevenue] + [ConversionGainsAndLossesOnForeignInvestments] + [GainLossOnSaleOfPreviouslyUnissuedStockBySubsidiaryOrEquityInvesteeNonoperatingIncome] - [MortgageServicingRightsMSRImpairmentRecovery] + [ForeignCurrencyTransactionGainLossBeforeTax] + [DebtInstrumentConvertibleBeneficialConversionFeature] + [GainsLossesOnExtinguishmentOfDebt]

## Interest Income/Expense

### GAAP

- InterestIncomeOperating = InterestIncomeOperatingPaidInCash + InterestIncomeOperatingPaidInKind
- InterestExpenseOperating = InterestExpenseDeposits + InterestExpenseTradingLiabilities + InterestExpenseBorrowings + InterestExpenseBeneficialInterestsIssuedByConsolidatedVariableInterestEntities + InterestExpenseTrustPreferredSecurities
- InterestIncomeExpenseNet = InterestAndFeeIncomeLoansAndLeases + InterestIncomeAndFeesBankersAcceptancesCertificatesOfDepositAndCommercialPaper + InterestIncomePurchasedReceivables + InterestIncomeDepositsWithFinancialInstitutions + InterestIncomeFederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell + InterestIncomeOperating - InterestExpenseOperating

## Revenue

- noninterest_income_operating = investment_banking_advisory_brokerage_underwriting_fees_commissions + gain_loss_sale_financial_assets + revenue_premiums + revenue_principal_transaction + noninterest_income_operating_other
- revenue_services_financial = investment_banking_advisory_brokerage_underwriting_fees_commissions + gain_loss_sale_financial_assets + revenue_premiums + revenue_principal_transaction + revenue_services_financial_other
- revenue_finance_excluding_interest_dividends
    - revenue_services_financial + revenue_market_data + noninterest_income_operating_other
    - noninterest_income_operating + revenue_market_data
- revenue_financial_noninterest = 
    - investment_banking_advisory_brokerage_underwriting_fees_commissions + revenue_premiums + revenue_principal_transaction + revenue_market_data + gain_loss_sale_financial_assets + noninterest_income_operating_other + revenue_services_financial_other
    - noninterest_income_operating + revenue_market_data + revenue_services_financial_other
    - revenue_finance_excluding_interest_dividends + revenue_market_data
- revenue_financial = revenue_financial_noninterest + revenue_financial_interest_dividend

### Old
- revenue_fees_commissions = revenue_fees_commissions_banking + revenue_commissions_brokerage + revenue_fees_servicing_financial_assets + revenue_fees_commissions_transfer_agent + revenue_fees_commissions_correspondent_clearing + revenue_fees_commissions_other + revenue_fees_commissions_insurance + revenue_fees_investment_advisory_management + revenue_fees_merchant_discount + revenue_fees_servicing_net
- investment_banking_advisory_brokerage_underwriting_fees_commissions = revenue_investment_banking + income_loss_underwriting + revenue_fees_commissions
- noninterest_income_operating = investment_banking_advisory_brokerage_underwriting_fees_commissions + revenue_principal_transaction + revenue_premiums + gain_loss_sale_financial_assets + noninterest_income_operating_other
- revenue_services_financial = revenue_fees_commissions + gain_loss_sale_financial_assets + revenue_principal_transaction + revenue_premiums + revenue_services_financial_other
- revenue_finance_excluding_interest_dividends = revenue_commissions_brokerage + revenue_investment_banking + income_loss_underwriting + revenue_principal_transaction + revenue_market_data
- revenue = 
    - revenue_sales + interest_dividend_income_operating + interest_income_expense_operating_after_provision_losses + noninterest_income_operating + revenue_finance_excluding_interest_dividends
    - revenue_contract + revenue_noncontract

### GAAP

+ IncreaseDecreaseInCarryingValueOfAssetsReceivedAsConsiderationInDisposalOfBusiness + OperatingLeasesIncomeStatementLeaseRevenue + ForeignCurrencyTransactionGainLossBeforeTax
- Revenues = 
    - RevenueFromContractWithCustomerIncludingAssessedTax + RevenueNotFromContractWithCustomer
    - InterestAndDividendIncomeOperating + RevenuesExcludingInterestAndDividends
    - InterestIncomeExpenseAfterProvisionForLoanLoss + NoninterestIncome
- SalesRevenueNet
- RevenuesExcludingInterestAndDividends =
  - BrokerageCommissionsRevenue + InvestmentBankingRevenue + UnderwritingIncomeLoss + PrincipalTransactionsRevenue + (FeesAndCommissions) + (InsuranceServicesRevenue) + MarketDataRevenue
- FinancialServicesRevenue = 
    - FeesAndCommissions + GainsLossesOnSalesOfAssets + PrincipalTransactionsRevenue + PremiumsEarnedNet + RevenueOtherFinancialServices
- InsuranceServicesRevenue = 
    - PremiumsEarnedNet + InsuranceInvestmentIncome + GainLossOnSaleOfInsuranceBlock + InsuranceAgencyManagementFee + InsuranceCommissionsAndFees

## Profit/Loss

- income_loss_continuing_operations_pretax_excluding_interest_expense = income_loss_operating + income_loss_nonoperating
- income_loss_continuing_operations_pretax_excluding_equity_method_investments = income_loss_continuing_operations_pretax_excluding_interest_expense - interest_debt_expense
- income_loss_continuing_operations_pretax = income_loss_continuing_operations_pretax_excluding_equity_method_investments + income_loss_equity_method_investment_pretax
- income_loss_continuing_operations_net_including_minority_interest = income_loss_continuing_operations_pretax - tax_income_expense_benefit
- income_loss_net_including_minority_interest = income_loss_continuing_operations_net_including_minority_interest + income_loss_discontinued_operation

- income_loss_excluding_extraordinary_items = income_loss_continued_operations_net + 

- income_loss_pretax_including_minority_interest = income_loss_continuing_operations_pretax + income_loss_discontinued_operations_pretax + extraordinary_items_gross


### GAAP

- IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments =
    - OperatingIncomeLoss + NonoperatingIncomeExpense - (InterestAndDebtExpense)
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic + IncomeLossFromContinuingOperationsBeforeIncomeTaxesForeign
- IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments
- IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments
- IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest = 
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments - IncomeTaxExpenseBenefit + IncomeLossFromEquityMethodInvestments
    - IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest - IncomeTaxExpenseBenefit
- IncomeLossFromContinuingOperations = IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest - IncomeLossFromContinuingOperationsAttributableToNoncontrollingEntity
- NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersBasic
- NetIncomeLossFromContinuingOperationsAvailableToCommonShareholdersDiluted
- IncomeLossBeforeGainOrLossOnSaleOfPropertiesExtraordinaryItemsAndCumulativeEffectsOfAccountingChanges = IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
- IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple = 
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
    - IncomeLossBeforeGainOrLossOnSaleOfPropertiesExtraordinaryItemsAndCumulativeEffectsOfAccountingChanges + GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes
- IncomeLossIncludingPortionAttributableToNoncontrollingInterest = IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments + IncomeLossFromEquityMethodInvestments + DiscontinuedOperationIncomeLossFromDiscontinuedOperationBeforeIncomeTax + ExtraordinaryItemsGross
- ProfitLoss = 
    - ExtraordinaryItemNetOfTax + IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple
    - NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest - NetIncomeLossAttributableToRedeemableNoncontrollingInterest + ExtraordinaryItemGainOrLossNetOfTaxAttributableToReportingEntity
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
    - InvestmentIncomeOperatingAfterExpenseAndTax + RealizedAndUnrealizedGainLossInvestmentAndDerivativeOperatingAfterTax + ForeignCurrencyTransactionGainLossAfterTax
- NetIncomeLoss = ProfitLoss - NetIncomeLossAttributableToNoncontrollingInterest
- NetIncomeLossAvailableToCommonStockholdersBasic = NetIncomeLoss - PreferredStockDividendsAndOtherAdjustments
- NetIncomeLossAvailableToCommonStockholdersDiluted = NetIncomeLossAvailableToCommonStockholdersBasic + InterestOnConvertibleDebtNetOfTax + ConvertiblePreferredDividendsNetOfTax + DilutiveSecurities

### IFRS
- ProfitLossFromContinuingOperations = ProfitLossBeforeTax - IncomeTaxExpenseContinuingOperations
- ProfitLoss = ProfitLossFromContinuingOperations + ProfitLossFromDiscontinuedOperations

### Nonoperating income

- NonoperatingGainsLosses = GainLossOnInvestments + DisposalGroupNotDiscontinuedOperationGainLossOnDisposal + GainLossOnSaleOfStockInSubsidiaryOrEquityMethodInvestee + DeconsolidationGainOrLossAmount + GainLossOnSaleOfPreviouslyUnissuedStockBySubsidiaryOrEquityInvesteeNonoperatingIncome + GainLossOnSaleOfInterestInProjects + GainLossOnDerivativeInstrumentsNetPretax + BusinessCombinationBargainPurchaseGainRecognizedAmount + SaleLeasebackTransactionCurrentPeriodGainRecognized + OtherNonoperatingGainsLosses
- NonoperatingIncomeExpense = 
    - NonoperatingGainsLosses + InvestmentIncomeNet + RoyaltyIncomeNonoperating + RentalIncomeNonoperating + DevelopmentProfitsNonoperating + LeveragedLeasesIncomeStatementNetIncomeFromLeveragedLeases + ForeignCurrencyTransactionGainLossBeforeTax + OtherNonoperatingAssetRelatedIncome + OtherNonoperatingIncomeExpense
    - InvestmentIncomeNonoperating + GainLossOnContractTermination + GainLossOnCondemnation - LossFromCatastrophes + PublicUtilitiesAllowanceForFundsUsedDuringConstructionAdditions + ForeignCurrencyTransactionGainLossBeforeTax - SalesTypeLeaseInitialDirectCostExpenseCommencement - OperatingLeaseInitialDirectCostExpenseOverTerm + GainLossOnSaleOfLeasedAssetsNetOperatingLeases + GainsLossesOnSalesOfOtherRealEstate + BankOwnedLifeInsuranceIncome + RealEstateInvestmentPartnershipRevenue + ConversionGainsAndLossesOnForeignInvestments + ProfitLossFromRealEstateOperations - MortgageServicingRightsMSRImpairmentRecovery + DebtInstrumentConvertibleBeneficialConversionFeature + PublicUtilitiesAllowanceForFundsUsedDuringConstructionCapitalizedCostOfEquity - NetPeriodicDefinedBenefitsExpenseReversalOfExpenseExcludingServiceCostComponent + GovernmentAssistanceNonoperatingIncome + GovernmentAssistanceNonoperatingExpense + OtherNonoperatingIncomeExpense - UnusualOrInfrequentItemNetGainLoss


## Cashflow

NetCashProvidedByUsedInOperatingActivitiesContinuingOperations = ProfitLoss - ExtraordinaryItemNetOfTax - IncomeLossFromDiscontinuedOperationsNetOfTax + AdjustmentsToReconcileNetIncomeLossToCashProvidedByUsedInOperatingActivities

ProfitLoss = 
    - ExtraordinaryItemNetOfTax + IncomeLossBeforeExtraordinaryItemsAndCumulativeEffectOfChangeInAccountingPrinciple
    - NetIncomeLossIncludingPortionAttributableToNonredeemableNoncontrollingInterest - NetIncomeLossAttributableToRedeemableNoncontrollingInterest + ExtraordinaryItemGainOrLossNetOfTaxAttributableToReportingEntity
    - IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest + IncomeLossFromDiscontinuedOperationsNetOfTax
    - InvestmentIncomeOperatingAfterExpenseAndTax + RealizedAndUnrealizedGainLossInvestmentAndDerivativeOperatingAfterTax + ForeignCurrencyTransactionGainLossAfterTax



## Asset valuation

- Carrying value/amount: book value
- Fair value: market value

Valuation adjustment:
- Accretion: unplanned increase in carrying value increase of a liability 
- Amortization: planned reduction in carrying value of intangible assets
- Depreciation: planned reducting in carrying value of tangible assets
- Impairment: unplanned reducting in carrying value of an asset (when fair value falls below carrying value)

Impairment of securities are classified as other than temporary (OTT) and are recognized either in earnings or comprehensive earnings.

### Marketable securities
- Available for Sale (AFS): fair value through other comprehensive income (FVOCI)
- Held to Maturity (HTM): carrying value at amortized cost basis
- Held for Trading (HFT): fair value through net income (FVNI)
